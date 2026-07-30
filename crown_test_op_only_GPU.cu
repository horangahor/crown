// crown_test_op.cu - 최적화된 GPU 버전 CROWN 테스트
// my_lirpa_op.cu 사용 (cuBLAS + 메모리 풀)
// 컴파일: nvcc -Xcompiler "/utf-8" -Xlinker "/STACK:134217728" -arch=sm_89 -lcublas crown_test_op.cu -o crown_test_op.exe

#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<chrono>
#include "my_lirpa_op_only_GPU.cu"
#include<fstream>

const char* MODEL_PATH = "C:/Users/user/Desktop/cuda/jnunnv_v1_0/jnunnv/models/Custom/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin";

// jnunnv_v1_0/CustomToLirpa.py 참조
// 모델.bin 의 구조
// 매직(4바이트)버전(4)가중치 레이어 개수(4) 레이어별 노드 수(4 * (m+1))
// 각 가중치 레이어의 가중치값 (n_out * n_in * 8바이트) / 편향값(n_out * 8바이트)
FullyConnectedNetwork* load_custom_network(const char* filepath){
    // 스택(Stack) 메모리 한계(기본 MSVC 1MB) 초과를 막기 위해 힙(Heap) 영역에 동적 할당
    FullyConnectedNetwork* net = new FullyConnectedNetwork();

    // 파일 오픈
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error opening file: " << filepath << std::endl;
        return net;
    }
    
    // magic Number 읽기
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "NNCB") {
        std::cerr << "Invalid custom binary (magic mismatch)" << std::endl;
        return net;
    }

    // 버전 읽기
    int version;
    file.read((char*)&version, sizeof(int));

    // 레이어 개수 읽기
    int m;
    file.read((char*)&m, sizeof(int));
    std::cout << "Magic: " << std::string(magic, 4) << ", Version: " << version << std::endl;
    std::cout << "Number of Weight Layers (m): " << m << std::endl;
    
    net->num_layers = m;

    // 레이어별 노드 수 읽기
    std::vector<int> sizes(m + 1);
    file.read((char*)sizes.data(), sizeof(int) * (m + 1));

    std::cout << "Number of each Layer Nodes (sizes): ";
    for(int i = 0; i < m + 1; ++i) {
        std::cout << sizes[i] << (i == m ? "" : ", ");
    }
    std::cout << std::endl;

    // 가중치(W)와 편향(b) 읽기
    for (int i = 0; i < m; ++i) {
        int n_in = sizes[i];
        int n_out = sizes[i + 1];

        net->layer_in_dim[i] = n_in;
        net->layer_out_dim[i] = n_out;

        int w_elements = n_out * n_in;
        int b_elements = n_out;

        std::vector<double> temp_W(w_elements);
        std::vector<double> temp_b(b_elements);

        file.read((char*)temp_W.data(), sizeof(double) * w_elements);
        file.read((char*)temp_b.data(), sizeof(double) * b_elements);

        std::cout << "- Layer " << i << " Weight read: " << n_out << " x " << n_in << std::endl;
        std::cout << "- Layer " << i << " Bias read: " << n_out << std::endl;

        net->W[i].rows = n_out;
        net->W[i].cols = n_in;
        for (int r = 0; r < n_out; ++r) {
            for (int c = 0; c < n_in; ++c) {
                net->W[i].a[r][c] = temp_W[r * n_in + c];
            }
        }

        net->b[i].n = n_out;
        for (int r = 0; r < n_out; ++r) {
            net->b[i].v[r] = temp_b[r];
        }

        // 활성화 함수 설정 (CustomToLirpa.py 규칙 적용)
        if (i < m - 1) {
            net->act[i] = ActivationType::Relu;
            std::cout << "  -> Activation: ReLU" << std::endl;
        } else {
            net->act[i] = ActivationType::Linear;
            std::cout << "  -> Activation: Linear" << std::endl;
        }
    }

    return net;
}

// 테스트 데이터(.bin)를 읽어오는 함수
Vector load_test_data(const char* filepath, int expected_dim) {
    Vector x0;
    x0.n = expected_dim;
    
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error opening data file: " << filepath << std::endl;
        return x0;
    }

    file.read((char*)x0.v, sizeof(double) * expected_dim);
    file.close();

    return x0;
}

int main(int argc, char** argv){

    const char* network_path = (argc > 1) ? argv[1] : MODEL_PATH;
    const char* data_path = (argc > 2) ? argv[2] : "test_data_1.bin";
    double eps = (argc > 3) ? std::stof(argv[3]) : 1e-6;
    
    std::cout << "=== OPTIMIZED GPU CROWN Test (cuBLAS + Memory Pool) ===" << std::endl;
    std::cout << "network read start " << std::endl;
    FullyConnectedNetwork* net = load_custom_network(network_path);
    std::cout << "network read success " << std::endl;

    // 준비된 테스트 데이터 1개 (bin 파일 로드)
    std::cout << "\ndata read start: " << data_path << std::endl;
    Vector x0 = load_test_data(data_path, net->layer_in_dim[0]);
    std::cout << "data read success (dim=" << x0.n << ")" << std::endl;

    // CROWN 알고리즘 호출 및 출력
    std::cout << "\n========================================\n";
    std::cout << "Starting CROWN Verification (eps = " << eps << ")" << std::endl;
    std::cout << "  ** MODE: GPU OPTIMIZED (cuBLAS + Pool) **" << std::endl;
    std::cout << "========================================\n";

    // 소수점 6자리까지 출력
    std::cout << std::fixed << std::setprecision(6);

    // 시간 측정 시작 (신경망 정방향 통과 및 CROWN)
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. 일반 예측값 확인 (신경망 정방향 통과)
    Vector y = network_forward(*net, x0);

    // 2. CROWN (Backward Bound) 호출
    BackwardBoundResult bwd = lirpa_backward_bound(*net, x0, eps);

    // 시간 측정 종료
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "calc success" << std::endl;
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;

    // 3. 출력 차원에 맞게 (보통 16개) 결과 출력
    int out_dim = bwd.final_lower.n;
    std::cout << "Output Dimension: " << out_dim << "\n\n";

    std::cout << "[Index] | Lower Bound | Upper Bound | Normal Pred (y)\n";
    std::cout << "---------------------------------------------------\n";
    for (int i = 0; i < out_dim; ++i) {
        std::cout << "[" << std::setw(3) << i << "]   |  " 
                  << std::setw(9) << bwd.final_lower.v[i] << "  |  "
                  << std::setw(9) << bwd.final_upper.v[i] << "  |  "
                  << std::setw(9) << y.v[i] << "\n";
    }
    std::cout << "========================================\n";

    // GPU 풀 해제
    gpu_pool_cleanup();

    // 동적 할당 해제
    delete net;
    return 0;
}
