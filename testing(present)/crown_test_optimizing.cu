#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<algorithm>
#include<chrono>
#include "my_lirpa_optimizing.cu"
#include<fstream>
#include <nvtx3/nvToolsExt.h>

const char* MODEL_PATH = "C:/Users/user/Desktop/cuda/jnunnv_v1_0/jnunnv/models/Custom/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin";

// jnunnv_v1_0/CustomToLirpa.py 참조
// 모델.bin 의 구조
// 매직(4바이트)버전(4)가중치 레이어 개수(4) 레이어별 노드 수(4 * (m+1))
// 각 가중치 레이어의 가중치값 (n_out * n_in * 8바이트) / 편향값(n_out * 8바이트)
FullyConnectedNetwork* load_custom_network(const char* filepath){
    // 스택(Stack) 메모리 한계(1MB) 초과를 막기 위해 힙(Heap) 영역에 동적 할당
    FullyConnectedNetwork* net = new FullyConnectedNetwork();

    // 파일 오픈
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error opening file: " << filepath << std::endl;
        return net;
    }
    
    // NNCB Magic Number 읽기
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "NNCB") {
        std::cerr << "유효한 커스텀 바이너리가 아닙니다. (magic 불일치)" << std::endl;
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
    // 가중치 레이어 수가 m 개 이므로 총 레이어 수는 m+1 개
    
    net->num_layers = m; // 신경망 구조체에 총 레이어 수 저장

    // 레이어별 노드 수 읽기 (sizes)
    // sizes는 (m + 1) 개의 정수(int) 배열
    // 예: 입력층 노드 수, 은닉층1 노드 수, ..., 출력층 노드 수
    std::vector<int> sizes(m + 1);
    // sizes.data() : 벡터가 가진 첫번째 데이터 주소 반환 , 배열처럼 쓰이므로
    // 벡터에는 4(바이트) / 4 / 4 / 4 /.... 이런식으로 각각 들어감 (4자리씩 m+1 개 읽으니까)
    file.read((char*)sizes.data(), sizeof(int) * (m + 1));

    std::cout << "Number of each Layer Nodes (sizes): ";
    for(int i = 0; i < m + 1; ++i) {
        std::cout << sizes[i] << (i == m ? "" : ", ");
    }
    std::cout << std::endl;

    // 가중치(W)와 편향(b) 읽기
    for (int i = 0; i < m; ++i) {
        // 특정 레이어의 입력 노드 수
        int n_in = sizes[i];
        // 특정 레이어의 출력 노드 수
        int n_out = sizes[i + 1];

        // 신경망 구조체에 in, out 차원 저장
        net->layer_in_dim[i] = n_in;
        net->layer_out_dim[i] = n_out;

        // 가중치의 개수는 (n_in x n_out) = (n_out x n_in)
        int w_elements = n_out * n_in;

        // 편향의 개수는 n_out
        int b_elements = n_out;

        // <f8 은 float64 ==> C++의 double
        std::vector<double> temp_W(w_elements);
        std::vector<double> temp_b(b_elements);

        file.read((char*)temp_W.data(), sizeof(double) * w_elements);
        file.read((char*)temp_b.data(), sizeof(double) * b_elements);

        std::cout << "- Layer " << i << " Weight read: " << n_out << " x " << n_in << std::endl;
        std::cout << "- Layer " << i << " Bias read: " << n_out << std::endl;
        
        // 1. 읽은 데이터를 FullyConnectedNetwork 구조체에 복사

        // 가중치는 행렬
        net->W[i].rows = n_out;
        net->W[i].cols = n_in;
        // 2차원 행렬에 각각 가중치값 집어넣기 <== device에서 하도록 가능
        for (int r = 0; r < n_out; ++r) {
            for (int c = 0; c < n_in; ++c) {
                // 바이너리(numpy row-major)에서 (r, c) 인덱스 찾기
                net->W[i].a[r][c] = temp_W[r * n_in + c];
            }
        }

        // 편향은 벡터
        net->b[i].n = n_out;
        // 편향 값 집어넣기 <== device에서 하도록 가능 
        for (int r = 0; r < n_out; ++r) {
            net->b[i].v[r] = temp_b[r];
        }

        // 2. 활성화 함수 설정 (CustomToLirpa.py 규칙 적용)
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

// 테스트 데이터(.bin)를 읽어오는 함수 (Pickle을 bin으로 만드는 모듈 readPickle.py 참고)
Vector load_test_data(const char* filepath, int expected_dim) {
    Vector x0;
    x0.n = expected_dim;
    
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error opening data file: " << filepath << std::endl;
        return x0;
    }

    // 64비트 실수(double) 배열로 통째로 읽기
    file.read((char*)x0.v, sizeof(double) * expected_dim);
    file.close();

    return x0;
}

int main(int argc, char** argv){

    const char* network_path = (argc > 1) ? argv[1] : MODEL_PATH;
    const char* data_path = (argc > 2) ? argv[2] : "test_data_1.bin";

    // atof , stof 둘 다 사용가능 
    // atof : 옛날 문자열 방식인 const char*를 바로 받음, 예외 처리시 0.0 반환
    // stof : 최신 문자열 방식(std::string)을 이용함, 예외 처리시 std::invalid_argument 반환 
    // 지금 argument 타입에 사용한 것은 char** 라서 argv[i]는 char* 타입
    // stof를 사용하면 string 객체를 만들어서 실수로 변환하는 미세한 오버헤드 발생 , 다만 확실한 예외처리 가능
    double eps = (argc > 3) ? std::stof(argv[3]) : 1e-6;
    
    std::cout << "network read start " << std::endl;
    // 포인터로 받음
    FullyConnectedNetwork* net = load_custom_network(network_path);
    std::cout << "network read success " << std::endl;

    // 준비된 테스트 데이터 1개 (bin 파일 로드)
    std::cout << "\ndata read start: " << data_path << std::endl;
    // 입력층 차원(net->layer_in_dim[0])만큼 읽어옴 (예: 256)
    Vector x0 = load_test_data(data_path, net->layer_in_dim[0]);
    std::cout << "data read success (dim=" << x0.n << ")" << std::endl;

    // CROWN 알고리즘 호출 및 출력
    std::cout << "\n========================================\n";
    std::cout << "Starting CROWN Verification (eps = " << eps << ") mempool + warmup" << std::endl;
    std::cout << "========================================\n";

    // 소수점 6자리까지 출력
    std::cout << std::fixed << std::setprecision(6);

    ensure_pool();
    // Immutable weights are uploaded and decomposed before timed verification.
    prepare_network_on_gpu(*net);

    // NVTX : 코드에 “이 구간이 실제 검증 시간이다”라는 표시를 넣는 기능
    nvtxRangePushA("CROWN_VERIFY");

    // 시간 측정 시작 (신경망 정방향 통과 및 CROWN)
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. 일반 예측값 확인 (신경망 정방향 통과)
    Vector y = network_forward(*net, x0);

    // 2. CROWN (Backward Bound) 호출
    // 여기서 선언시 BackwardBoundResult 빈 공간(4MB 할당 + 자잘한 벡터들까지 해서 넉넉히 0.25MB )
    // 내부 함수에 진입해서 보면 거의 64MB가 필요함 (함수 주석 참고) <== 근데 64MB 안되서 128MB로 함
    // 왜인지는 모르겠다 너무 함수가 복잡해져서..
    BackwardBoundResult bwd = lirpa_backward_bound(*net, x0, eps, false);

    // 시간 측정 종료
    auto t_end = std::chrono::high_resolution_clock::now();

    // NVTX 구간 종료
    nvtxRangePop();
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

    // 노드 결과 구조체
    struct NodeResult {
        int index;
        double lower;
        double upper;
        double pred;
        double range;
    };

    // 구조체를 element 로 하는 벡터
    std::vector<NodeResult> results;
    for (int i = 0; i < out_dim; ++i) {
        results.push_back({i, bwd.final_lower.v[i], bwd.final_upper.v[i], y.v[i], bwd.final_upper.v[i] - bwd.final_lower.v[i]});
    }

    int print_count = std::min(8, out_dim);

    // 신경망 출력값(Normal Pred) 정렬
    std::sort(results.begin(), results.end(), [](const NodeResult& a, const NodeResult& b) {
        return a.pred > b.pred;
    });

    std::cout << "\n[8 Nodes - Sorted by Normal Pred]\n";
    std::cout << "[Index] | Lower Bound | Upper Bound | Normal Pred (y) | Range (Upper - Lower)\n";
    std::cout << "-----------------------------------------------------------------------\n";
    for (int i = 0; i < print_count; ++i) {
        const auto& r = results[i];
        std::cout << "[" << std::setw(3) << r.index << "]   |  " 
                  << std::setw(9) << r.lower << "  |  "
                  << std::setw(9) << r.upper << "  |  "
                  << std::setw(9) << r.pred << "      |  "
                  << std::setw(9) << r.range << "\n";
    }
    std::cout << "=======================================================================\n";

    // 신경망 정렬 (범위)
    std::sort(results.begin(), results.end(), [](const NodeResult& a, const NodeResult& b) {
        return a.range < b.range;
    });

    std::cout << "\n[8 Nodes - Sorted by Range]\n";
    std::cout << "[Index] | Lower Bound | Upper Bound | Normal Pred (y) | Range (Upper - Lower)\n";
    std::cout << "-----------------------------------------------------------------------\n";
    for (int i = 0; i < print_count; ++i) {
        const auto& r = results[i];
        std::cout << "[" << std::setw(3) << r.index << "]   |  " 
                  << std::setw(9) << r.lower << "  |  "
                  << std::setw(9) << r.upper << "  |  "
                  << std::setw(9) << r.pred << "      |  "
                  << std::setw(9) << r.range << "\n";
    }
    std::cout << "=======================================================================\n";

    // 동적 할당 해제
    delete net;

    //system("pause");
    return 0;
}
