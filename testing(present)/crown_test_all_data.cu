#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<algorithm>
#include<numeric>
#include<chrono>
#include "my_lirpa_optimizing.cu"
#include<fstream>
#include <nvtx3/nvToolsExt.h>

const char* MODEL_PATH = "C:/Users/user/Desktop/cuda/crown_sparse/Custom/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin";

// 컴파일 : nvcc -Xcompiler "/utf-8" -Xlinker "/STACK:134217728" -arch=sm_89 crown_test_all_data.cu -o crown_test_all_data.exe

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

        // .bin 파일은 float64(double) 형식으로 저장됨 -> double로 읽고 float으로 변환
        std::vector<double> temp_W(w_elements);
        std::vector<double> temp_b(b_elements);

        file.read((char*)temp_W.data(), sizeof(double) * w_elements);
        file.read((char*)temp_b.data(), sizeof(double) * b_elements);

        std::cout << "- Layer " << i << " Weight read: " << n_out << " x " << n_in << std::endl;
        std::cout << "- Layer " << i << " Bias read: " << n_out << std::endl;
        
        // 가중치는 행렬
        net->W[i].rows = n_out;
        net->W[i].cols = n_in;
        // double -> float 변환하여 저장 (FP32 전환의 핵심)
        for (int r = 0; r < n_out; ++r) {
            for (int c = 0; c < n_in; ++c) {
                net->W[i].a[r][c] = static_cast<float>(temp_W[r * n_in + c]);
            }
        }

        // 편향은 벡터
        net->b[i].n = n_out;
        for (int r = 0; r < n_out; ++r) {
            net->b[i].v[r] = static_cast<float>(temp_b[r]);
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

// 전체 데이터셋 로더: (N x input_dim) float64 이진 파일을 한번에 로드
// 변환 방법 (convert_to_bin.py): X.astype("<f8").tofile("test_data_all.bin")
std::vector<Vector> load_all_data(const char* filepath, int input_dim, int& N_out) {
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening data file: " << filepath << std::endl;
        N_out = 0; return {};
    }
    // 파일 크기로 총 샘플 수 자동 계산
    file.seekg(0, std::ios::end);
    long long file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    int N = static_cast<int>(file_size / (sizeof(double) * input_dim));
    N_out = N;
    std::cout << "Data file size: " << file_size << " bytes" << std::endl;
    std::cout << "Detected N = " << N << " samples (dim=" << input_dim << ")" << std::endl;
    std::vector<Vector> dataset(N);
    std::vector<double> temp(input_dim);
    for (int i = 0; i < N; ++i) {
        file.read((char*)temp.data(), sizeof(double) * input_dim);
        dataset[i].n = input_dim;
        for (int d = 0; d < input_dim; ++d)
            dataset[i].v[d] = static_cast<float>(temp[d]);
    }
    file.close();
    return dataset;
}

// Top-k 강건성 인증 판정
// sparse.py certify_topk_selection 함수 기반 구현 (max_diff CROWN과 매치되는 신경망 선택 안테나의 개수가 k-max_diff개 이상이면 봐줌(true))
// 신경망 추론 값 : 신경망이 추론하고 선택한 상위 8개의 안테나
// CROWN : 입력값에 대해 eps 만큼의 노이즈를 주었을 때 해당 신경망에서 나올 수 있는 출력값의 범위가 수학적으로 증명되어 있음 
// (즉 lower bound 정렬로 8개 선택 가능시 해당 입력값 +- eps 내에서 수학적으로 상위 8개의 안테나가 항상 유지 된다는 소리)
// CROWN 에서 노드 8개를 선택 가능한 경우(4번 과정), 신경망이 추론해 정렬한 결과와 CROWN lower bound의 인덱스가 매치하면 신경망이 같은 안테나를 선택했으므로
// 실제 환경에서 해당 신경망의 입력에 noise가 추가되었을 때, 정답 8개 (== CROWN이 내놓은 상위 8개와 동일하다, 실험 때 lower bound 정렬이 noise 추가 안된 추론값의 정렬과 같으므로) 
bool certify_topk(const Vector& y0, const Vector& lb, const Vector& ub, int k = 8, int max_diff = 0) {
    int out_dim = y0.n;
    // 1. clean_topk: y0 오름차순 정렬, 뒤 k개 = top-k
    std::vector<int> y_order(out_dim);
    std::iota(y_order.begin(), y_order.end(), 0); // 0 ~ 15 까지 연속된 숫자를 채워줌 (index 반환용)
    std::sort(y_order.begin(), y_order.end(),
              [&](int a, int b){ return y0.v[a] < y0.v[b]; }); // 0~ 15번 노드의 추론 값, 인덱스가 매치 된 상황에서 오름차순 정렬
    // 2. order_lb: lb 내림차순 정렬
    std::vector<int> order_lb(out_dim);
    std::iota(order_lb.begin(), order_lb.end(), 0); // 0 ~ 15 까지 연속된 숫자를 채워줌 (index 반환용)
    std::sort(order_lb.begin(), order_lb.end(),
              [&](int a, int b){ return lb.v[a] > lb.v[b]; }); // 0~ 15번 노드의 CROWN 하한값, 인덱스가 매치 된 상황에서 내림차순 정렬
    // 3. sorted_ub: ub 내림차순 정렬
    std::vector<float> sorted_ub(out_dim);
    for (int i = 0; i < out_dim; ++i) sorted_ub[i] = ub.v[i];
    std::sort(sorted_ub.begin(), sorted_ub.end(), std::greater<float>()); // 0~ 15번 노드의 CROWN 상한값, 인덱스가 매치 된 상황에서 내림차순 정렬
    // 4. gap_ok: lb[order_lb[k-1]] > sorted_ub[k]
    if (lb.v[order_lb[k - 1]] <= sorted_ub[k]) return false;
    // 5. guaranteed_topk (order_lb 앞 k개) 와 clean_topk 교집합 == k
    int match = 0;
    for (int i = 0; i < k; ++i)    // 0 ~ 7
        for (int j = out_dim - k; j < out_dim; ++j)   // 16 - 8 = 8 , 8 ~ 15
            if (order_lb[i] == y_order[j]) { ++match; break; } // 가장 높은 lower bound 인덱스 8개랑 , 상위 8개의 추론값인 y0 인덱스가 몇개 포함되는지 구함
    return (match => k - max_diff); // 8개 다 일치하면 해당 입력에 대해서는 강건성이 보장되었단 소리(epsilon으로 인한 CROWN 결과와 출력 값의 상위 8개 선택이 일치) true
}
// 즉 eps 를 바꿔가며 T/F 비율을 구해서 해당 신경망이 어디 eps까지 버틸 수 있는지 비율을 구하는 것임 (비율을 보고 관리자가 판단)

int main(int argc, char** argv) {

    const char* network_path = (argc > 1) ? argv[1] : MODEL_PATH;
    const char* data_path    = (argc > 2) ? argv[2] : "test_data_all.bin";
    int         k            = (argc > 3) ? std::atoi(argv[3]) : 8;
    int         method       = (argc > 4) ? std::atoi(argv[4]) : 0; // 0: forward, 1: backward, // 추가 ? ==> 2: backward_only
    const char* output_csv   = (argc > 5) ? argv[5] : "results_cuda.csv";

    // 1. 모델 로드
    std::cout << "\n=== crown_test_all_data ===" << std::endl;
    std::cout << "network read start" << std::endl;
    FullyConnectedNetwork* net = load_custom_network(network_path);
    std::cout << "network read success" << std::endl;

    // 2. GPU 메모리 풀 초기화 + 불변 가중치 GPU 업로드
    ensure_pool();
    prepare_network_on_gpu(*net);
    std::cout << "GPU pool and weights ready" << std::endl;

    // 3. 전체 데이터셋 로드
    int N = 0;
    std::cout << "\ndata read start: " << data_path << std::endl;
    std::vector<Vector> dataset = load_all_data(data_path, net->layer_in_dim[0], N);
    if (N == 0) {
        std::cerr << "데이터를 읽지 못했습니다." << std::endl;
        delete net; return 1;
    }
    std::cout << "loaded " << N << " points (dim=" << net->layer_in_dim[0] << ")" << std::endl;

    // 4. eps 리스트 (sparse.py와 동일)
    const std::vector<double> eps_list = {
        1e-6, //1e-5,
        //1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
        //1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
        //1e-2, 1e-1, 1.0
    };

    struct EpsResult { int T, F; double ratio; };
    std::vector<EpsResult> all_results;

    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting CROWN Sweep" << std::endl;
    std::cout << "  data points : " << N               << std::endl;
    std::cout << "  k (top-k)   : " << k               << std::endl;
    std::cout << "  eps values  : " << eps_list.size() << std::endl;
    std::cout << "========================================\n" << std::endl;

    // 5. 전체 시간 측정 시작
    auto total_start = std::chrono::high_resolution_clock::now();
    nvtxRangePushA("CROWN_SWEEP");

    // 6. eps x 데이터 이중 루프 (sparse.py sweep() 와 완전히 동일)
    for (double eps : eps_list) {
        int t_count = 0, f_count = 0;
        float eps_f = static_cast<float>(eps);

        for (int i = 0; i < N; ++i) {
            // 신경망 정방향 통과
            Vector y0 = network_forward(*net, dataset[i]);
            // CROWN backward bound or forward bound 계산
            BackwardBoundResult bwd = lirpa_backward_bound(*net, dataset[i], eps_f, false, false);
            // ForwardBoundResult bwd = lirpa_forward_bound_impl(*net, dataset[i], eps_f, false, false, true);
            
            // Top-k 인증 판정
            if (certify_topk(y0, bwd.final_lower, bwd.final_upper, k))
                ++t_count;
            else
                ++f_count;

            // 1000개마다 진행상황 출력 (sparse.py 와 동일 포맷)
            if ((i + 1) % 1000 == 0) {
                std::cout << "  eps=" << std::setw(10)
                          << std::scientific << std::setprecision(1) << eps
                          << ": " << (i + 1) << "/" << N
                          << " points done (T=" << t_count
                          << " F=" << f_count << ")" << std::endl;
                std::cout.flush();
            }
        }
        int total = t_count + f_count;
        double ratio = (total > 0) ? static_cast<double>(t_count) / total : 0.0;
        all_results.push_back({t_count, f_count, ratio});
    }
    nvtxRangePop();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(total_end - total_start).count();

    // 7. 최종 결과 출력 (sparse.py 마지막 출력과 동일 포맷)
    std::cout << std::endl;
    for (int i = 0; i < (int)eps_list.size(); ++i) {
        std::cout << "eps=" << std::setw(10)
                  << std::scientific << std::setprecision(1) << eps_list[i]
                  << std::fixed
                  << ": T=" << std::setw(5) << all_results[i].T
                  << " F=" << std::setw(5) << all_results[i].F
                  << " ratio=" << std::setprecision(3) << all_results[i].ratio << std::endl;
    }
    std::cout << "\ntotal elapsed: "
              << std::fixed << std::setprecision(2) << total_elapsed << "s" << std::endl;

    // 8. CSV 저장 (sparse.py와 동일 컬럼: method, eps, T, F, ratio)
    std::ofstream csv_file(output_csv);
    if (csv_file.is_open()) {
        csv_file << "method,eps,T,F,ratio" << std::endl;
        for (int i = 0; i < (int)eps_list.size(); ++i) {
            csv_file << "cuda_gpu,"
                     << std::scientific << std::setprecision(6) << eps_list[i] << ","
                     << all_results[i].T << ","
                     << all_results[i].F << ","
                     << std::fixed << std::setprecision(6) << all_results[i].ratio << std::endl;
        }
        csv_file.close();
        std::cout << "results saved to " << output_csv << std::endl;
    } else {
        std::cerr << "Error: could not write to " << output_csv << std::endl;
    }

    // 9. 메모리 해제
    delete net;
    return 0;
}
