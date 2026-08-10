// crown_test_op_hybrid_v2_Sparse.cu
#include<iostream>
#include<cuda_runtime.h>
#include<vector>
#include<chrono>
#include "my_lirpa_op_hybrid_v2_Sparse.cu"
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

int main(int argc, char** argv){
    const char* network_path = (argc > 1) ? argv[1] : MODEL_PATH;
    const char* data_path = (argc > 2) ? argv[2] : "test_data_1.bin";
    double eps = (argc > 3) ? std::stof(argv[3]) : 1e-6;
    
    std::cout << "=== HYBRID V2 CROWN Test (Sparse) ===" << std::endl;
    
    // GPU & cuSPARSE warmup
    warmup_cusparse();

    std::cout << "network read start " << std::endl;
    FullyConnectedNetwork* net = load_custom_network(network_path);
    std::cout << "network read success " << std::endl;

    // 가중치 GPU 사전 로드 (Dense -> Sparse 변환)
    std::cout << "uploading weights to GPU as CSR..." << std::endl;
    for (int i = 0; i < net->num_layers; ++i) {
        net->W_sparse[i] = dense_to_csr(net->W[i]);
        net->W_pos_sparse[i] = dense_to_csr(positive_part(net->W[i]));
        net->W_neg_sparse[i] = dense_to_csr(negative_part(net->W[i]));
    }
    std::cout << "upload success" << std::endl;

    std::cout << "\ndata read start: " << data_path << std::endl;
    Vector x0 = load_test_data(data_path, net->layer_in_dim[0]);
    std::cout << "data read success (dim=" << x0.n << ")" << std::endl;

    std::cout << "\n========================================\n";
    std::cout << "Starting CROWN Verification (eps = " << eps << ")" << std::endl;
    std::cout << "  ** MODE: HYBRID V2 SPARSE **" << std::endl;
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(6);

    auto t_start = std::chrono::high_resolution_clock::now();

    Vector y;
    BackwardBoundResult bwd;
    const int NUM_ITERS = 1; // 원하시면 500으로 변경 가능
    
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        y = network_forward(*net, x0);
        bwd = lirpa_backward_bound(*net, x0, eps);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << "calc success " << NUM_ITERS << "iterations" << std::endl;
    std::cout << "Total Elapsed time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Average time per iter: " << elapsed_ms / NUM_ITERS << " ms" << std::endl;

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

    for (int i = 0; i < net->num_layers; ++i) {
        net->W_sparse[i].destroy();
        net->W_pos_sparse[i].destroy();
        net->W_neg_sparse[i].destroy();
    }
    gpu_pool_cleanup();
    delete net;
    return 0;
}
