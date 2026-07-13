#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// using namespace std; 안 쓴 이유 : 이름 충돌을 막기 위함, 코드가 복잡해지면 std:: 을 일일이 붙여야함
// using namespace std 붙인것과 명시적으로 std:: 를 사용하는 것에는 성능 차이가 없음 <== 컴파일러에게 알려주는 것이므로 컴파일 타임에는 미세하게 변화가 있을 수도 (std 명시쪽이 미세하게 더 빠름?)

// 익명 namespace, 같은 파일 내에서만 접근 가능 <== 여기서만 쓰는 새로운 구조체 등을 정의
namespace {

    int MAX_LAYERS = 16;
    int MAX_DIM = 64;

    // 항상 최대의 행렬을 가지지만 rows, cols 정보에 의해 크기가 제한되는 형식
    struct Matrix{
        int rows = 0;
        int cols = 0;
        double a[MAX_DIM][MAX_DIM]{}; // 유니폼 초기화, 0.0으로 초기화됨
    };

    // 크기가 n인 벡터
    struct Vector{
        int n = 0;
        double v[MAX_DIM]{}; // 유니폼 초기화, 0.0으로 초기화됨
    };

    // 설명 필요
    struct AffineBound {
        Matrix lower_A;
        Vector lower_c;
        Matrix upper_A;
        Vector upper_c;
    };

    // 설명 필요
    struct LayerBound {
        int dim = 0;
        Vector alpha_lower;
        Vector beta_lower;
        Vector alpha_upper;
        Vector beta_upper;
    };

    // 설명 필요
    struct ForwardBoundResult {
        AffineBound final_affine;
        Vector final_lower;
        Vector final_upper;
        int num_layer_bounds = 0;
        LayerBound layer_bounds[MAX_LAYERS]{};
    };

    // 설명 필요
    struct BackwardBoundResult {
        AffineBound final_affine;
        Vector final_lower;
        Vector final_upper;
        int num_layer_bounds = 0;
        LayerBound layer_bounds[MAX_LAYERS]{};
    };

    // 설명 필요
    struct FullyConnectedNetwork {
        int num_layers = 0;
        int layer_in_dim[MAX_LAYERS]{};
        int layer_out_dim[MAX_LAYERS]{};
        Matrix W[MAX_LAYERS]{};
        Vector b[MAX_LAYERS]{};
        ActivationType act[MAX_LAYERS]{};
    };

    // 인라인 함수, macro 처럼 코드 자체를 치환함 (실행 시간을 최적화 하기 위함)
    inline double pos(double x) {
    return std::max(x, 0.0);
    }

    inline double neg(double x) {
    return std::min(x, 0.0);
    }

    // relu
    inline double relu(double x) {
    return std::max(x, 0.0);
    }

    // sigmoid
    inline double sigmoid(double x) {
        if (x >= 0.0) {
            return 1.0 / (1.0 + std::exp(-x));
        }
        const double ex = std::exp(x);
        return ex / (1.0 + ex);
    }

    // sigmoid prime
    inline double sigmoid_prime(double x) {
        const double s = sigmoid(x);
        return s * (1.0 - s);
    }

    // cond, msg를 인자로 받아 cond 가 true 이면 넘어가고, false 이면 msg 를 출력하며 에러 발생
    // string& 으로 하면 매개변수로 전달될 때 복사가 일어나지 않으므로(기존 메모리를 참조함) 속도가 빠름
    void require(bool cond, const string& msg) {
        if (!cond) {
            throw std::invalid_argument(msg);
        }
    }

    // 인자를 입력받아 크기를 지정한 2차원 0 행렬 만들기
    Matrix make_zero_matrix(int rows, int cols) {
        require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM, "Matrix shape out of bounds.");
        Matrix out;
        out.rows = rows;
        out.cols = cols;
        return out;
    }

    // 인자를 입력받아 1차원 0 벡터 배열 만들기
    Vector make_zero_vector(int n) {
        require(n >= 0 && n <= MAX_DIM, "Vector length out of bounds.");
        Vector out;
        out.n = n;
        return out;
    }

    // 크기가 n * n 인 단위행렬 생성
    Matrix make_eye(int n) {
        Matrix out = make_zero_matrix(n, n);
        for (int i = 0; i < n; ++i) {
            out.a[i][i] = 1.0;
        }
        return out;
    }

}




// argc 전달된 인자의 개수 (실행될 프로그램도 인자로 포함되므로 무조건 기본값 1을 가짐)
// argv 인수의 문자열 배열(포인터)

// ex. argv[0] : 프로그램 이름(기본값), argv[1] : 첫번째 추가 인수 ... 
int main(int argc, char** argv) {
    try {
        double eps = 0.02; // 엡실론 기본 값
        if (argc >= 2){
            eps = stod(argv[1]); // 엡실론 = 추가 인수 // stod 문자열 -> 실수 변환
        }

        // todo 



    }



}


/*
int main(int argc, char** argv) {
    try {
        double eps = 0.02;
        if (argc >= 2) {
            eps = std::stod(argv[1]);
        }

        self_test_relaxations();
        run_xor_demo(eps);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}




*/