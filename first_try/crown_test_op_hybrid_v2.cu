// crown_test_op_hybrid_v2.cu - 하이브리드 최적화 V2 테스트
// 1. cuBLAS 웜업
// 2. 가중치 GPU 사전 할당 및 복사
// 3. cudaMemcpy2D로 실제 필요 크기만 복사
// 컴파일: nvcc -Xcompiler "/utf-8" -Xlinker "/STACK:134217728" -arch=sm_89 -lcublas crown_test_op_hybrid_v2.cu -o crown_test_op_hybrid_v2.exe

#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<chrono>
#include "my_lirpa_op_hybrid_v2.cu"
#include<fstream>

const char* MODEL_PATH = "C:/Users/user/Desktop/cuda/jnunnv_v1_0/jnunnv/models/Custom/Baseline mMIMO FC H hard short 80 HTHNN_LAY2_491 RELU 20241018 PRUNED 0.93_NO_SIGMOID_custom.bin";

FullyConnectedNetwork* load_custom_network(const char* filepath){
    FullyConnectedNetwork* net = new FullyConnectedNetwork();

    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if(!file.is_open()){
        std::cerr << "Error opening file: " << filepath << std::endl;
        return net;
    }
    
    char magic[4];
    file.read(magic, 4);
    if (std::string(magic, 4) != "NNCB") {
        std::cerr << "Invalid custom binary (magic mismatch)" << std::endl;
        return net;
    }

    int version;
    file.read((char*)&version, sizeof(int));

    int m;
    file.read((char*)&m, sizeof(int));
    std::cout << "Magic: " << std::string(magic, 4) << ", Version: " << version << std::endl;
    std::cout << "Number of Weight Layers (m): " << m << std::endl;
    
    net->num_layers = m;

    std::vector<int> sizes(m + 1);
    file.read((char*)sizes.data(), sizeof(int) * (m + 1));

    std::cout << "Layer sizes: ";
    for(int i = 0; i < m + 1; ++i)
        std::cout << sizes[i] << (i == m ? "" : ", ");
    std::cout << std::endl;

    for (int i = 0; i < m; ++i) {
        int n_in = sizes[i];
        int n_out = sizes[i + 1];
        net->layer_in_dim[i] = n_in;
        net->layer_out_dim[i] = n_out;

        std::vector<double> temp_W(n_out * n_in);
        std::vector<double> temp_b(n_out);
        file.read((char*)temp_W.data(), sizeof(double) * n_out * n_in);
        file.read((char*)temp_b.data(), sizeof(double) * n_out);

        std::cout << "- Layer " << i << " Weight: " << n_out << " x " << n_in << std::endl;

        net->W[i].rows = n_out;
        net->W[i].cols = n_in;
        for (int r = 0; r < n_out; ++r)
            for (int c = 0; c < n_in; ++c)
                net->W[i].a[r][c] = temp_W[r * n_in + c];

        net->b[i].n = n_out;
        for (int r = 0; r < n_out; ++r)
            net->b[i].v[r] = temp_b[r];

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

// 가중치를 GPU에 사전 로드
void pre_upload_weights_to_gpu(FullyConnectedNetwork* net) {
    for (int i = 0; i < net->num_layers; ++i) {
        cudaMalloc(&net->d_W[i], MAX_DIM * MAX_DIM * sizeof(double));
        cudaMemcpy2D(net->d_W[i], MAX_DIM * sizeof(double),
                     net->W[i].a, MAX_DIM * sizeof(double),
                     net->W[i].cols * sizeof(double), net->W[i].rows,
                     cudaMemcpyHostToDevice);
    }
}

void free_weights_from_gpu(FullyConnectedNetwork* net) {
    for (int i = 0; i < net->num_layers; ++i) {
        if (net->d_W[i]) cudaFree(net->d_W[i]);
    }
}

int main(int argc, char** argv){
    const char* network_path = (argc > 1) ? argv[1] : MODEL_PATH;
    const char* data_path = (argc > 2) ? argv[2] : "test_data_1.bin";
    double eps = (argc > 3) ? std::stof(argv[3]) : 1e-6;
    
    std::cout << "=== HYBRID V2 CROWN Test (Warmup, Memcpy2D, Preload) ===" << std::endl;
    
    // GPU & cuBLAS 웜업
    warmup_cublas();

    std::cout << "network read start " << std::endl;
    FullyConnectedNetwork* net = load_custom_network(network_path);
    std::cout << "network read success " << std::endl;

    // 가중치 GPU 업로드
    pre_upload_weights_to_gpu(net);

    std::cout << "\ndata read start: " << data_path << std::endl;
    Vector x0 = load_test_data(data_path, net->layer_in_dim[0]);
    std::cout << "data read success (dim=" << x0.n << ")" << std::endl;

    std::cout << "\n========================================\n";
    std::cout << "Starting CROWN Verification (eps = " << eps << ")" << std::endl;
    std::cout << "  ** MODE: HYBRID V2 **" << std::endl;
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(6);

    auto t_start = std::chrono::high_resolution_clock::now();

    Vector y = network_forward(*net, x0);
    BackwardBoundResult bwd = lirpa_backward_bound(*net, x0, eps);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "calc success" << std::endl;
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;

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

    free_weights_from_gpu(net);
    gpu_pool_cleanup();
    delete net;
    return 0;
}
