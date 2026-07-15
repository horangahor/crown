#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// using namespace std; 안 쓴 이유 : 이름 충돌을 막기 위함, 코드가 복잡해지면
// std:: 을 일일이 붙여야함 using namespace std 붙인것과 명시적으로 std:: 를
// 사용하는 것에는 성능 차이가 없음 <== 컴파일러에게 알려주는 것이므로 컴파일
// 타임에는 미세하게 변화가 있을 수도 (std 명시쪽이 미세하게 더 빠름?)

// 익명 namespace, 같은 파일 내에서만 접근 가능 <== 여기서만 쓰는 새로운 구조체
// 등을 정의
namespace {

constexpr int MAX_LAYERS = 16;
constexpr int MAX_DIM = 64;

enum class ActivationType {
  Relu,
  Sigmoid,
  Linear,
};

// 항상 최대의 행렬을 가지지만 rows, cols 정보에 의해 크기가 제한되는 형식
struct Matrix {
  int rows = 0;
  int cols = 0;
  double a[MAX_DIM][MAX_DIM]{}; // 유니폼 초기화, 0.0으로 초기화됨
};

// 크기가 n인 벡터
struct Vector {
  int n = 0;
  double v[MAX_DIM]{}; // 유니폼 초기화, 0.0으로 초기화됨
};

// 선형 바운드
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
__host__ __device__ inline double pos(double x) { return x > 0.0 ? x : 0.0; }

__host__ __device__ inline double neg(double x) { return x < 0.0 ? x : 0.0; }

// relu
__host__ __device__ inline double relu(double x) { return x > 0.0 ? x : 0.0; }

// sigmoid
__host__ __device__ inline double sigmoid(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + exp(-x));
  }
  const double ex = exp(x);
  return ex / (1.0 + ex);
}

// sigmoid prime
__host__ __device__ inline double sigmoid_prime(double x) {
  const double s = sigmoid(x);
  return s * (1.0 - s);
}

// cond, msg를 인자로 받아 cond 가 true 이면 넘어가고, false 이면 msg 를
// 출력하며 에러 발생 string& 으로 하면 매개변수로 전달될 때 복사가 일어나지
// 않으므로(기존 메모리를 참조함) 속도가 빠름
void require(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument(msg);
  }
}

// 인자를 입력받아 크기를 지정한 2차원 0 행렬 만들기
Matrix make_zero_matrix(int rows, int cols) {
  require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM,
          "Matrix shape out of bounds.");
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

// 1. GPU 커널 함수: GPU 스레드, 행렬 더하기
// 2. CPU 커널 호출 함수: 할당 -> 복사 -> 연산지시 -> 회수 -> 반납

// 행렬 합 구하기
__global__ void mat_add_gpu(const Matrix *A, const Matrix *B, Matrix *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;

  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    C->a[r][c] = A->a[r][c] + B->a[r][c];
  }
}

Matrix mat_add(const Matrix &A, const Matrix &B) {
  require(A.rows == B.rows && A.cols == B.cols, "mat_add shape mismatch.");

  Matrix C = make_zero_matrix(A.rows, A.cols);

  // GPU 메모리 포인터
  Matrix *d_A, *d_B, *d_C;

  // 할당 (cudaMalloc)
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_B, sizeof(Matrix));
  cudaMalloc((void **)&d_C, sizeof(Matrix));

  // CPU -> GPU (cudaMemcpy)
  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, &B, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  // 스레드 분배
  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  // 커널 실행 (GPU 연산 지시)
  mat_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaDeviceSynchronize(); // 완료 대기

  // 결과 회수: GPU -> CPU (cudaMemcpy)
  cudaMemcpy(&C, d_C, sizeof(Matrix), cudaMemcpyDeviceToHost);

  // 해제 (cudaFree)
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}

// 벡터 합
__global__ void vec_add_gpu(const Vector *A, const Vector *B, Vector *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->n;

  if (tid < total_elements) {
    C->v[tid] = A->v[tid] + B->v[tid];
  }
}

Vector vec_add(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_add shape mismatch.");

  Vector c = make_zero_vector(a.n);

  Vector *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(Vector));
  cudaMalloc((void **)&d_b, sizeof(Vector));
  cudaMalloc((void **)&d_c, sizeof(Vector));

  cudaMemcpy(d_a, &a, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int total_elements = a.n;
  int threadsPerBlock = 256;

  // 블록 개수 계산
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  vec_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);

  cudaDeviceSynchronize();

  cudaMemcpy(&c, d_c, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}

// 벡터 빼기
__global__ void vec_sub_gpu(const Vector *A, const Vector *B, Vector *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->n;

  if (tid < total_elements) {
    C->v[tid] = A->v[tid] - B->v[tid];
  }
}

Vector vec_sub(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_sub shape mismatch.");

  Vector c = make_zero_vector(a.n);

  Vector *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(Vector));
  cudaMalloc((void **)&d_b, sizeof(Vector));
  cudaMalloc((void **)&d_c, sizeof(Vector));

  cudaMemcpy(d_a, &a, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  vec_sub_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();

  cudaMemcpy(&c, d_c, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}

// matmul, 행렬 곱셈
// 각 스레드가 결과 행렬의 원소 하나를 담당, 내적 계산
__global__ void matmul_gpu(const Matrix *A, const Matrix *B, Matrix *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * B->cols;

  if (tid < total_elements) {
    int r = tid / B->cols;
    int c = tid % B->cols;
    double sum = 0.0;
    for (int k = 0; k < A->cols; ++k) {
      if (A->a[r][k] == 0.0) {
        continue;
      }
      sum += A->a[r][k] * B->a[k][c];
    }
    C->a[r][c] = sum;
  }
}

Matrix matmul(const Matrix &A, const Matrix &B) {
  require(A.cols == B.rows, "matmul shape mismatch.");

  Matrix C = make_zero_matrix(A.rows, B.cols);

  Matrix *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_B, sizeof(Matrix));
  cudaMalloc((void **)&d_C, sizeof(Matrix));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, &B, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * B.cols; // 행렬곱의 크기
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
  cudaDeviceSynchronize();

  cudaMemcpy(&C, d_C, sizeof(Matrix), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}

// matvec, 행렬-벡터 곱
// 각 스레드가 결과 벡터의 원소 하나를 담당
__global__ void matvec_gpu(const Matrix *A, const Vector *x, Vector *y) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < A->rows) {
    double sum = 0.0;
    for (int j = 0; j < A->cols; ++j) {
      sum += A->a[tid][j] * x->v[j];
    }
    y->v[tid] = sum;
  }
}

Vector matvec(const Matrix &A, const Vector &x) {
  require(A.cols == x.n, "matvec shape mismatch.");

  Vector y = make_zero_vector(A.rows);

  Matrix *d_A;
  Vector *d_x, *d_y;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_x, sizeof(Vector));
  cudaMalloc((void **)&d_y, sizeof(Vector));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, &x, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, &y, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  matvec_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_x, d_y);
  cudaDeviceSynchronize();

  cudaMemcpy(&y, d_y, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);

  return y;
}

// positive_part, 양수부 추출
__global__ void positive_part_gpu(const Matrix *A, Matrix *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;

  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    double val = A->a[r][c];
    out->a[r][c] = pos(val);
  }
}

Matrix positive_part(const Matrix &A) {
  Matrix out = make_zero_matrix(A.rows, A.cols);

  Matrix *d_A, *d_out;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_out, sizeof(Matrix));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  positive_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, d_out, sizeof(Matrix), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_out);

  return out;
}

// negative_part, 음수부 추출
__global__ void negative_part_gpu(const Matrix *A, Matrix *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;

  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    double val = A->a[r][c];
    out->a[r][c] = neg(val);
  }
}

Matrix negative_part(const Matrix &A) {
  Matrix out = make_zero_matrix(A.rows, A.cols);

  Matrix *d_A, *d_out;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_out, sizeof(Matrix));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  negative_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, d_out, sizeof(Matrix), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_out);

  return out;
}

// rowwise_scale, 행별 스칼라 곱
// A의 i번째 행 전체에 s[i]를 곱함
__global__ void rowwise_scale_gpu(const Matrix *A, const Vector *s,
                                  Matrix *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;

  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    out->a[r][c] = A->a[r][c] * s->v[r];
  }
}

Matrix rowwise_scale(const Matrix &A, const Vector &s) {
  require(A.rows == s.n, "rowwise_scale shape mismatch.");

  Matrix out = make_zero_matrix(A.rows, A.cols);

  Matrix *d_A, *d_out;
  Vector *d_s;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_s, sizeof(Vector));
  cudaMalloc((void **)&d_out, sizeof(Matrix));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_s, &s, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  rowwise_scale_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_s, d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, d_out, sizeof(Matrix), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_s);
  cudaFree(d_out);

  return out;
}

// elemwise_mul, 벡터 원소별 곱
__global__ void elemwise_mul_gpu(const Vector *A, const Vector *B, Vector *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < A->n) {
    C->v[tid] = A->v[tid] * B->v[tid];
  }
}

Vector elemwise_mul(const Vector &a, const Vector &b) {
  require(a.n == b.n, "elemwise_mul shape mismatch.");

  Vector c = make_zero_vector(a.n);

  Vector *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, sizeof(Vector));
  cudaMalloc((void **)&d_b, sizeof(Vector));
  cudaMalloc((void **)&d_c, sizeof(Vector));

  cudaMemcpy(d_a, &a, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  elemwise_mul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();

  cudaMemcpy(&c, d_c, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}

// 엡실론 벡터 만들기 (n 크기의) , 단순히 eps 로 채운 벡터이므로 gpu 연산 불필요
Vector make_eps_vec(int n, double eps) {
  Vector out = make_zero_vector(n);
  for (int i = 0; i < n; ++i) {
    out.v[i] = eps;
  }
  return out;
}

// affine_min
__global__ void affine_min_gpu(const Matrix *A, const Vector *c,
                               const Vector *xl, const Vector *xu,
                               Vector *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < A->rows) {
    double sum = c->v[tid];
    for (int j = 0; j < A->cols; ++j) {
      double aij = A->a[tid][j];
      sum += pos(aij) * xl->v[j] + neg(aij) * xu->v[j];
    }
    out->v[tid] = sum;
  }
}

Vector affine_min(const Matrix &A, const Vector &c, const Vector &x0,
                  double eps) {
  require(A.rows == c.n && A.cols == x0.n, "affine_min shape mismatch.");

  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);
  Vector out = make_zero_vector(A.rows);

  Matrix *d_A;
  Vector *d_c, *d_xl, *d_xu, *d_out;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_c, sizeof(Vector));
  cudaMalloc((void **)&d_xl, sizeof(Vector));
  cudaMalloc((void **)&d_xu, sizeof(Vector));
  cudaMalloc((void **)&d_out, sizeof(Vector));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xl, &xl, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xu, &xu, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  affine_min_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_c, d_xl, d_xu,
                                                     d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, d_out, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_c);
  cudaFree(d_xl);
  cudaFree(d_xu);
  cudaFree(d_out);

  return out;
}

// affine_max
__global__ void affine_max_gpu(const Matrix *A, const Vector *c,
                               const Vector *xl, const Vector *xu,
                               Vector *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < A->rows) {
    double sum = c->v[tid];
    for (int j = 0; j < A->cols; ++j) {
      double aij = A->a[tid][j];
      sum += pos(aij) * xu->v[j] + neg(aij) * xl->v[j];
    }
    out->v[tid] = sum;
  }
}

Vector affine_max(const Matrix &A, const Vector &c, const Vector &x0,
                  double eps) {
  require(A.rows == c.n && A.cols == x0.n, "affine_max shape mismatch.");

  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);
  Vector out = make_zero_vector(A.rows);

  Matrix *d_A;
  Vector *d_c, *d_xl, *d_xu, *d_out;
  cudaMalloc((void **)&d_A, sizeof(Matrix));
  cudaMalloc((void **)&d_c, sizeof(Vector));
  cudaMalloc((void **)&d_xl, sizeof(Vector));
  cudaMalloc((void **)&d_xu, sizeof(Vector));
  cudaMalloc((void **)&d_out, sizeof(Vector));

  cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, &c, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xl, &xl, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xu, &xu, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  affine_max_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_c, d_xl, d_xu,
                                                     d_out);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, d_out, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_c);
  cudaFree(d_xl);
  cudaFree(d_xu);
  cudaFree(d_out);

  return out;
}

// relu_relax
__global__ void relu_relax_gpu(const Vector *lower, const Vector *upper,
                               Vector *alpha_l, Vector *beta_l, Vector *alpha_u,
                               Vector *beta_u) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < lower->n) {
    double l = lower->v[tid];
    double u = upper->v[tid];
    assert(l <= u); // 기존 relu_relax 에 l > u 감지

    if (l >= 0.0) {
      alpha_l->v[tid] = 1.0;
      alpha_u->v[tid] = 1.0;
    } else if (u <= 0.0) {
      // keep zeros
    } else {
      double denom = u - l;
      alpha_u->v[tid] = u / denom;
      beta_u->v[tid] = -u * l / denom;

      // CUDA에서는 std::abs 대신 fabs 사용
      bool use_identity_lower = fabs(l) < fabs(u);
      alpha_l->v[tid] = use_identity_lower ? 1.0 : 0.0;
      beta_l->v[tid] = 0.0;
    }
  }
}

void relu_relax(const Vector &lower, const Vector &upper, Vector &alpha_l,
                Vector &beta_l, Vector &alpha_u, Vector &beta_u) {
  require(lower.n == upper.n, "relu_relax shape mismatch.");

  alpha_l = make_zero_vector(lower.n);
  beta_l = make_zero_vector(lower.n);
  alpha_u = make_zero_vector(lower.n);
  beta_u = make_zero_vector(lower.n);

  Vector *d_lower, *d_upper, *d_alpha_l, *d_beta_l, *d_alpha_u, *d_beta_u;
  cudaMalloc((void **)&d_lower, sizeof(Vector));
  cudaMalloc((void **)&d_upper, sizeof(Vector));
  cudaMalloc((void **)&d_alpha_l, sizeof(Vector));
  cudaMalloc((void **)&d_beta_l, sizeof(Vector));
  cudaMalloc((void **)&d_alpha_u, sizeof(Vector));
  cudaMalloc((void **)&d_beta_u, sizeof(Vector));

  cudaMemcpy(d_lower, &lower, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_upper, &upper, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alpha_l, &alpha_l, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta_l, &beta_l, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_alpha_u, &alpha_u, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta_u, &beta_u, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (lower.n + threadsPerBlock - 1) / threadsPerBlock;

  relu_relax_gpu<<<blocksPerGrid, threadsPerBlock>>>(
      d_lower, d_upper, d_alpha_l, d_beta_l, d_alpha_u, d_beta_u);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in relu_relax_gpu launch: "
              << cudaGetErrorString(err) << "\n";
  }

  // 커널 내의 assert() 실패는 이 동기화 함수에서 에러로 잡힘
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error during relu_relax_gpu execution: "
              << cudaGetErrorString(err) << "\n";
    throw std::runtime_error("relu_relax_gpu 실행 중 에러 발생");
  }

  err = cudaMemcpy(&alpha_l, d_alpha_l, sizeof(Vector), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in relu_relax memcpy back: "
              << cudaGetErrorString(err) << "\n";
  }
  cudaMemcpy(&beta_l, d_beta_l, sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(&alpha_u, d_alpha_u, sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(&beta_u, d_beta_u, sizeof(Vector), cudaMemcpyDeviceToHost);

  cudaFree(d_lower);
  cudaFree(d_upper);
  cudaFree(d_alpha_l);
  cudaFree(d_beta_l);
  cudaFree(d_alpha_u);
  cudaFree(d_beta_u);
}

double bisect_root(double lo, double hi,
                   const std::function<double(double)> &fn, int max_iter = 80,
                   double tol = 1e-12) {
  double flo = fn(lo);
  double fhi = fn(hi);

  if (std::abs(flo) < tol)
    return lo;
  if (std::abs(fhi) < tol)
    return hi;

  if (flo * fhi > 0.0) {
    constexpr int GRID = 257;
    std::array<double, GRID> xs{};
    std::array<double, GRID> vals{};
    for (int i = 0; i < GRID; ++i) {
      xs[i] = lo + (hi - lo) * static_cast<double>(i) /
                       static_cast<double>(GRID - 1);
      vals[i] = fn(xs[i]);
    }

    int best = 0;
    double best_abs = std::abs(vals[0]);
    for (int i = 1; i < GRID; ++i) {
      const double cur = std::abs(vals[i]);
      if (cur < best_abs) {
        best_abs = cur;
        best = i;
      }
    }

    bool found = false;
    for (int i = 0; i < GRID - 1; ++i) {
      if (vals[i] == 0.0 || vals[i] * vals[i + 1] <= 0.0) {
        lo = xs[i];
        hi = xs[i + 1];
        flo = vals[i];
        found = true;
        break;
      }
    }
    if (!found) {
      return xs[best];
    }
  }

  for (int it = 0; it < max_iter; ++it) {
    const double mid = 0.5 * (lo + hi);
    const double fmid = fn(mid);
    if (std::abs(fmid) < tol || std::abs(hi - lo) < tol) {
      return mid;
    }
    if (flo * fmid <= 0.0) {
      hi = mid;
    } else {
      lo = mid;
      flo = fmid;
    }
  }
  return 0.5 * (lo + hi);
}

void sigmoid_relax(const Vector &lower, const Vector &upper, Vector &alpha_l,
                   Vector &beta_l, Vector &alpha_u, Vector &beta_u) {
  require(lower.n == upper.n, "sigmoid_relax shape mismatch.");
  alpha_l = make_zero_vector(lower.n);
  beta_l = make_zero_vector(lower.n);
  alpha_u = make_zero_vector(lower.n);
  beta_u = make_zero_vector(lower.n);

  for (int i = 0; i < lower.n; ++i) {
    const double l = lower.v[i];
    const double u = upper.v[i];
    require(l <= u, "Invalid interval in sigmoid_relax.");

    if (std::abs(u - l) < 1e-14) {
      const double slope = sigmoid_prime(l);
      const double intercept = sigmoid(l) - slope * l;
      alpha_l.v[i] = slope;
      beta_l.v[i] = intercept;
      alpha_u.v[i] = slope;
      beta_u.v[i] = intercept;
      continue;
    }

    if (l >= 0.0) {
      const double slope_sec = (sigmoid(u) - sigmoid(l)) / (u - l);
      alpha_l.v[i] = slope_sec;
      beta_l.v[i] = sigmoid(u) - slope_sec * u;

      const double x0 = 0.5 * (l + u);
      const double slope_tan = sigmoid_prime(x0);
      alpha_u.v[i] = slope_tan;
      beta_u.v[i] = sigmoid(x0) - slope_tan * x0;
    } else if (u <= 0.0) {
      const double x0 = 0.5 * (l + u);
      const double slope_tan = sigmoid_prime(x0);
      alpha_l.v[i] = slope_tan;
      beta_l.v[i] = sigmoid(x0) - slope_tan * x0;

      const double slope_sec = (sigmoid(u) - sigmoid(l)) / (u - l);
      alpha_u.v[i] = slope_sec;
      beta_u.v[i] = sigmoid(u) - slope_sec * u;
    } else {
      const double su = sigmoid(u);
      const auto fn_lower = [su, u](double d) {
        return (su - sigmoid(d)) / (u - d) - sigmoid_prime(d);
      };
      const double du = bisect_root(l, 0.0, fn_lower);

      const double sl = sigmoid(l);
      const auto fn_upper = [sl, l](double d) {
        return (sigmoid(d) - sl) / (d - l) - sigmoid_prime(d);
      };
      const double dl = bisect_root(0.0, u, fn_upper);

      const double slope_lower = sigmoid_prime(du);
      alpha_l.v[i] = slope_lower;
      beta_l.v[i] = sigmoid(du) - slope_lower * du;

      const double slope_upper = sigmoid_prime(dl);
      alpha_u.v[i] = slope_upper;
      beta_u.v[i] = sigmoid(dl) - slope_upper * dl;
    }

    double lower_violation = 0.0;
    double upper_violation = 0.0;
    constexpr int SAMPLES = 1001;
    for (int k = 0; k < SAMPLES; ++k) {
      const double x = l + (u - l) * static_cast<double>(k) /
                               static_cast<double>(SAMPLES - 1);
      const double y = sigmoid(x);
      const double lower_line = alpha_l.v[i] * x + beta_l.v[i];
      const double upper_line = alpha_u.v[i] * x + beta_u.v[i];
      lower_violation = std::max(lower_violation, lower_line - y);
      upper_violation = std::max(upper_violation, y - upper_line);
    }
    if (lower_violation > 1e-10) {
      beta_l.v[i] -= lower_violation + 1e-10;
    }
    if (upper_violation > 1e-10) {
      beta_u.v[i] += upper_violation + 1e-10;
    }
  }
}

ActivationType parse_activation(const std::string &name) {
  std::string lower = name;
  for (char &ch : lower) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (lower == "relu") {
    return ActivationType::Relu;
  }
  if (lower == "sigmoid") {
    return ActivationType::Sigmoid;
  }
  if (lower == "linear") {
    return ActivationType::Linear;
  }
  throw std::invalid_argument("Unsupported activation: " + name);
}

FullyConnectedNetwork
make_network(int num_layers, const int *layer_in_dim, const int *layer_out_dim,
             const double W_data[MAX_LAYERS][MAX_DIM][MAX_DIM],
             const double b_data[MAX_LAYERS][MAX_DIM],
             const std::string *activations) {
  require(num_layers >= 1 && num_layers <= MAX_LAYERS,
          "num_layers out of bounds.");

  FullyConnectedNetwork net;
  net.num_layers = num_layers;

  for (int l = 0; l < num_layers; ++l) {
    const int in_d = layer_in_dim[l];
    const int out_d = layer_out_dim[l];
    require(in_d >= 1 && in_d <= MAX_DIM, "layer_in_dim out of bounds.");
    require(out_d >= 1 && out_d <= MAX_DIM, "layer_out_dim out of bounds.");

    net.layer_in_dim[l] = in_d;
    net.layer_out_dim[l] = out_d;
    net.W[l] = make_zero_matrix(out_d, in_d);
    net.b[l] = make_zero_vector(out_d);
    net.act[l] = parse_activation(activations[l]);

    for (int i = 0; i < out_d; ++i) {
      net.b[l].v[i] = b_data[l][i];
      for (int j = 0; j < in_d; ++j) {
        net.W[l].a[i][j] = W_data[l][i][j];
      }
    }

    if (l > 0) {
      require(net.layer_out_dim[l - 1] == in_d, "Layer dimension mismatch.");
    }
  }

  return net;
}

int network_input_dim(const FullyConnectedNetwork &net) {
  return net.layer_in_dim[0];
}

int network_output_dim(const FullyConnectedNetwork &net) {
  return net.layer_out_dim[net.num_layers - 1];
}

Vector network_forward(const FullyConnectedNetwork &net, const Vector &x) {
  require(x.n == network_input_dim(net),
          "network_forward input dimension mismatch.");
  Vector f = x;

  for (int l = 0; l < net.num_layers; ++l) {
    Vector s = vec_add(matvec(net.W[l], f), net.b[l]);
    for (int i = 0; i < s.n; ++i) {
      if (net.act[l] == ActivationType::Relu) {
        s.v[i] = relu(s.v[i]);
      } else if (net.act[l] == ActivationType::Sigmoid) {
        s.v[i] = sigmoid(s.v[i]);
      } else if (net.act[l] == ActivationType::Linear) {
        // unchanged
      }
    }
    f = s;
  }

  return f;
}

ForwardBoundResult lirpa_forward_bound(const FullyConnectedNetwork &net,
                                       const Vector &x0, double eps) {
  require(x0.n == network_input_dim(net),
          "lirpa_forward_bound input dimension mismatch.");

  const int in_dim = network_input_dim(net);
  AffineBound current;
  current.lower_A = make_eye(in_dim);
  current.upper_A = make_eye(in_dim);
  current.lower_c = make_zero_vector(in_dim);
  current.upper_c = make_zero_vector(in_dim);

  ForwardBoundResult out;
  out.num_layer_bounds = net.num_layers;

  for (int l = 0; l < net.num_layers; ++l) {
    const Matrix W_pos = positive_part(net.W[l]);
    const Matrix W_neg = negative_part(net.W[l]);

    AffineBound pre;
    pre.lower_A =
        mat_add(matmul(W_pos, current.lower_A), matmul(W_neg, current.upper_A));
    pre.lower_c = vec_add(
        vec_add(matvec(W_pos, current.lower_c), matvec(W_neg, current.upper_c)),
        net.b[l]);

    pre.upper_A =
        mat_add(matmul(W_pos, current.upper_A), matmul(W_neg, current.lower_A));
    pre.upper_c = vec_add(
        vec_add(matvec(W_pos, current.upper_c), matvec(W_neg, current.lower_c)),
        net.b[l]);

    const Vector pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
    const Vector pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

    Vector alpha_l, beta_l, alpha_u, beta_u;
    if (net.act[l] == ActivationType::Linear) {
      alpha_l = make_zero_vector(pre_lower.n);
      alpha_u = make_zero_vector(pre_lower.n);
      beta_l = make_zero_vector(pre_lower.n);
      beta_u = make_zero_vector(pre_lower.n);
      for (int i = 0; i < pre_lower.n; ++i) {
        alpha_l.v[i] = 1.0;
        alpha_u.v[i] = 1.0;
      }
    } else if (net.act[l] == ActivationType::Relu) {
      relu_relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    } else {
      sigmoid_relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    }

    AffineBound post;
    post.lower_A = rowwise_scale(pre.lower_A, alpha_l);
    post.lower_c = vec_add(elemwise_mul(alpha_l, pre.lower_c), beta_l);
    post.upper_A = rowwise_scale(pre.upper_A, alpha_u);
    post.upper_c = vec_add(elemwise_mul(alpha_u, pre.upper_c), beta_u);

    current = post;

    out.layer_bounds[l].dim = alpha_l.n;
    out.layer_bounds[l].alpha_lower = alpha_l;
    out.layer_bounds[l].beta_lower = beta_l;
    out.layer_bounds[l].alpha_upper = alpha_u;
    out.layer_bounds[l].beta_upper = beta_u;
  }

  out.final_affine = current;
  out.final_lower = affine_min(current.lower_A, current.lower_c, x0, eps);
  out.final_upper = affine_max(current.upper_A, current.upper_c, x0, eps);
  return out;
}

void backward_one_layer(Matrix &lower_M, Vector &lower_p, Matrix &upper_M,
                        Vector &upper_p, const Matrix &W, const Vector &b,
                        const Vector &alpha_l, const Vector &beta_l,
                        const Vector &alpha_u, const Vector &beta_u) {
  require(lower_M.cols == W.rows && upper_M.cols == W.rows,
          "backward_one_layer shape mismatch.");
  require(alpha_l.n == W.rows && beta_l.n == W.rows && alpha_u.n == W.rows &&
              beta_u.n == W.rows,
          "backward_one_layer relaxation shape mismatch.");

  const Matrix lower_M_pos = positive_part(lower_M);
  const Matrix lower_M_neg = negative_part(lower_M);
  const Matrix upper_M_pos = positive_part(upper_M);
  const Matrix upper_M_neg = negative_part(upper_M);

  Matrix lower_s_coeff = make_zero_matrix(lower_M.rows, lower_M.cols);
  Matrix upper_s_coeff = make_zero_matrix(upper_M.rows, upper_M.cols);
  for (int i = 0; i < lower_M.rows; ++i) {
    for (int j = 0; j < lower_M.cols; ++j) {
      lower_s_coeff.a[i][j] = lower_M_pos.a[i][j] * alpha_l.v[j] +
                              lower_M_neg.a[i][j] * alpha_u.v[j];
      upper_s_coeff.a[i][j] = upper_M_pos.a[i][j] * alpha_u.v[j] +
                              upper_M_neg.a[i][j] * alpha_l.v[j];
    }
  }

  const Matrix new_lower_M = matmul(lower_s_coeff, W);
  const Matrix new_upper_M = matmul(upper_s_coeff, W);

  const Vector term_lp = vec_add(elemwise_mul(alpha_l, b), beta_l);
  const Vector term_ln = vec_add(elemwise_mul(alpha_u, b), beta_u);
  const Vector new_lower_p = vec_add(
      vec_add(matvec(lower_M_pos, term_lp), matvec(lower_M_neg, term_ln)),
      lower_p);

  const Vector term_up = vec_add(elemwise_mul(alpha_u, b), beta_u);
  const Vector term_un = vec_add(elemwise_mul(alpha_l, b), beta_l);
  const Vector new_upper_p = vec_add(
      vec_add(matvec(upper_M_pos, term_up), matvec(upper_M_neg, term_un)),
      upper_p);

  lower_M = new_lower_M;
  lower_p = new_lower_p;
  upper_M = new_upper_M;
  upper_p = new_upper_p;
}

BackwardBoundResult
lirpa_backward_bound(const FullyConnectedNetwork &net, const Vector &x0,
                     double eps, const Matrix *output_lower_M = nullptr,
                     const Vector *output_lower_p = nullptr,
                     const Matrix *output_upper_M = nullptr,
                     const Vector *output_upper_p = nullptr) {
  const ForwardBoundResult fwd = lirpa_forward_bound(net, x0, eps);

  const int output_dim = network_output_dim(net);
  Matrix lower_M;
  Matrix upper_M;
  Vector lower_p;
  Vector upper_p;

  if (output_lower_M) {
    lower_M = *output_lower_M;
  } else {
    lower_M = make_eye(output_dim);
  }
  if (output_upper_M) {
    upper_M = *output_upper_M;
  } else {
    upper_M = make_eye(output_dim);
  }

  if (output_lower_p) {
    lower_p = *output_lower_p;
  } else {
    lower_p = make_zero_vector(lower_M.rows);
  }
  if (output_upper_p) {
    upper_p = *output_upper_p;
  } else {
    upper_p = make_zero_vector(upper_M.rows);
  }

  require(lower_M.cols == output_dim && upper_M.cols == output_dim,
          "Output spec matrix column mismatch.");
  require(lower_M.rows == upper_M.rows,
          "Output spec lower/upper row mismatch.");
  require(lower_p.n == lower_M.rows && upper_p.n == upper_M.rows,
          "Output spec vector row mismatch.");

  for (int l = net.num_layers - 1; l >= 0; --l) {
    const LayerBound &lb = fwd.layer_bounds[l];
    backward_one_layer(lower_M, lower_p, upper_M, upper_p, net.W[l], net.b[l],
                       lb.alpha_lower, lb.beta_lower, lb.alpha_upper,
                       lb.beta_upper);
  }

  BackwardBoundResult out;
  out.final_affine.lower_A = lower_M;
  out.final_affine.lower_c = lower_p;
  out.final_affine.upper_A = upper_M;
  out.final_affine.upper_c = upper_p;
  out.final_lower = affine_min(lower_M, lower_p, x0, eps);
  out.final_upper = affine_max(upper_M, upper_p, x0, eps);
  out.num_layer_bounds = fwd.num_layer_bounds;
  for (int i = 0; i < fwd.num_layer_bounds; ++i) {
    out.layer_bounds[i] = fwd.layer_bounds[i];
  }
  return out;
}

class LiRPABackwardOnly {
public:
  BackwardBoundResult bound(const FullyConnectedNetwork &net, const Vector &x0,
                            double eps, const Matrix *output_lower_M = nullptr,
                            const Vector *output_lower_p = nullptr,
                            const Matrix *output_upper_M = nullptr,
                            const Vector *output_upper_p = nullptr) const {
    require(x0.n == network_input_dim(net),
            "LiRPABackwardOnly input dimension mismatch.");

    LayerBound layer_bounds[MAX_LAYERS]{};
    for (int l = 0; l < net.num_layers; ++l) {
      layer_bounds[l] =
          build_one_layer_relaxation(net, l, x0, eps, layer_bounds);
    }

    const int output_dim = network_output_dim(net);
    Matrix lower_M = output_lower_M ? *output_lower_M : make_eye(output_dim);
    Matrix upper_M = output_upper_M ? *output_upper_M : make_eye(output_dim);
    Vector lower_p =
        output_lower_p ? *output_lower_p : make_zero_vector(lower_M.rows);
    Vector upper_p =
        output_upper_p ? *output_upper_p : make_zero_vector(upper_M.rows);

    require(lower_M.cols == output_dim && upper_M.cols == output_dim,
            "Output spec matrix column mismatch.");
    require(lower_M.rows == upper_M.rows,
            "Output spec lower/upper row mismatch.");
    require(lower_p.n == lower_M.rows && upper_p.n == upper_M.rows,
            "Output spec vector row mismatch.");

    for (int l = net.num_layers - 1; l >= 0; --l) {
      const LayerBound &lb = layer_bounds[l];
      backward_one_layer(lower_M, lower_p, upper_M, upper_p, net.W[l], net.b[l],
                         lb.alpha_lower, lb.beta_lower, lb.alpha_upper,
                         lb.beta_upper);
    }

    BackwardBoundResult out;
    out.final_affine.lower_A = lower_M;
    out.final_affine.lower_c = lower_p;
    out.final_affine.upper_A = upper_M;
    out.final_affine.upper_c = upper_p;
    out.final_lower = affine_min(lower_M, lower_p, x0, eps);
    out.final_upper = affine_max(upper_M, upper_p, x0, eps);
    out.num_layer_bounds = net.num_layers;
    for (int l = 0; l < net.num_layers; ++l) {
      out.layer_bounds[l] = layer_bounds[l];
    }
    return out;
  }

private:
  static void relax_activation(ActivationType act, const Vector &pre_lower,
                               const Vector &pre_upper, Vector &alpha_l,
                               Vector &beta_l, Vector &alpha_u,
                               Vector &beta_u) {
    if (act == ActivationType::Linear) {
      alpha_l = make_zero_vector(pre_lower.n);
      alpha_u = make_zero_vector(pre_lower.n);
      beta_l = make_zero_vector(pre_lower.n);
      beta_u = make_zero_vector(pre_lower.n);
      for (int i = 0; i < pre_lower.n; ++i) {
        alpha_l.v[i] = 1.0;
        alpha_u.v[i] = 1.0;
      }
    } else if (act == ActivationType::Relu) {
      relu_relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    } else if (act == ActivationType::Sigmoid) {
      sigmoid_relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    } else {
      throw std::invalid_argument("No relaxation for activation.");
    }
  }

  static LayerBound
  build_one_layer_relaxation(const FullyConnectedNetwork &net, int layer,
                             const Vector &x0, double eps,
                             const LayerBound *previous_layer_bounds) {
    Matrix lower_M = net.W[layer];
    Matrix upper_M = net.W[layer];
    Vector lower_p = net.b[layer];
    Vector upper_p = net.b[layer];

    for (int prev = layer - 1; prev >= 0; --prev) {
      const LayerBound &lb = previous_layer_bounds[prev];
      backward_one_layer(lower_M, lower_p, upper_M, upper_p, net.W[prev],
                         net.b[prev], lb.alpha_lower, lb.beta_lower,
                         lb.alpha_upper, lb.beta_upper);
    }

    const Vector pre_lower = affine_min(lower_M, lower_p, x0, eps);
    const Vector pre_upper = affine_max(upper_M, upper_p, x0, eps);

    Vector alpha_l, beta_l, alpha_u, beta_u;
    relax_activation(net.act[layer], pre_lower, pre_upper, alpha_l, beta_l,
                     alpha_u, beta_u);

    LayerBound out;
    out.dim = alpha_l.n;
    out.alpha_lower = alpha_l;
    out.beta_lower = beta_l;
    out.alpha_upper = alpha_u;
    out.beta_upper = beta_u;
    return out;
  }
};

void self_test_relaxations() {
  std::mt19937_64 rng(0);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  for (int t = 0; t < 200; ++t) {
    double a = dist(rng);
    double b = dist(rng);
    if (a > b) {
      std::swap(a, b);
    }
    if (std::abs(a - b) < 1e-8) {
      b = a + 1e-6;
    }

    Vector l = make_zero_vector(1);
    Vector u = make_zero_vector(1);
    l.v[0] = a;
    u.v[0] = b;

    Vector al, bl, au, bu;

    relu_relax(l, u, al, bl, au, bu);
    for (int k = 0; k < 201; ++k) {
      const double x = a + (b - a) * static_cast<double>(k) / 200.0;
      const double y = relu(x);
      const double lhs = al.v[0] * x + bl.v[0];
      const double rhs = au.v[0] * x + bu.v[0];
      if (lhs > y + 1e-8 || y > rhs + 1e-8) {
        std::cerr << "ReLU test failed! a=" << a << ", b=" << b << ", x=" << x
                  << ", y=" << y << ", al=" << al.v[0] << ", bl=" << bl.v[0]
                  << ", au=" << au.v[0] << ", bu=" << bu.v[0] << "\n";
        std::cerr << "lhs=" << lhs << ", rhs=" << rhs << "\n";
        throw std::runtime_error("ReLU relaxation self-test failed.");
      }
    }

    sigmoid_relax(l, u, al, bl, au, bu);
    for (int k = 0; k < 201; ++k) {
      const double x = a + (b - a) * static_cast<double>(k) / 200.0;
      const double y = sigmoid(x);
      const double lhs = al.v[0] * x + bl.v[0];
      const double rhs = au.v[0] * x + bu.v[0];
      if (lhs > y + 1e-8 || y > rhs + 1e-8) {
        throw std::runtime_error("Sigmoid relaxation self-test failed.");
      }
    }
  }
}

FullyConnectedNetwork make_xor_network() {
  int layer_in[MAX_LAYERS]{};
  int layer_out[MAX_LAYERS]{};
  double W_data[MAX_LAYERS][MAX_DIM][MAX_DIM]{};
  double b_data[MAX_LAYERS][MAX_DIM]{};
  std::string acts[MAX_LAYERS];

  const int L = 2;
  layer_in[0] = 2;
  layer_out[0] = 2;
  acts[0] = "relu";

  layer_in[1] = 2;
  layer_out[1] = 1;
  acts[1] = "sigmoid";

  W_data[0][0][0] = 2.1247;
  W_data[0][0][1] = 2.1267;
  W_data[0][1][0] = -2.1237;
  W_data[0][1][1] = -2.1235;
  b_data[0][0] = -2.1259;
  b_data[0][1] = 2.1234;

  W_data[1][0][0] = -3.6788;
  W_data[1][0][1] = -3.6766;
  b_data[1][0] = 3.5451;

  return make_network(L, layer_in, layer_out, W_data, b_data, acts);
}

int xor_expected_label(const Vector &x) {
  require(x.n >= 2,
          "xor_expected_label requires at least 2-dimensional input.");
  const int a = static_cast<int>(std::llround(x.v[0]));
  const int b = static_cast<int>(std::llround(x.v[1]));
  return a ^ b;
}

void run_xor_demo(double eps) {
  const FullyConnectedNetwork network = make_xor_network();
  const LiRPABackwardOnly backward_only_verifier;

  Vector points[4];
  for (int i = 0; i < 4; ++i) {
    points[i] = make_zero_vector(2);
  }
  points[0].v[0] = 0.0;
  points[0].v[1] = 0.0;
  points[1].v[0] = 0.0;
  points[1].v[1] = 1.0;
  points[2].v[0] = 1.0;
  points[2].v[1] = 0.0;
  points[3].v[0] = 1.0;
  points[3].v[1] = 1.0;

  std::cout
      << "XOR network point predictions and LiRPA-certified output bounds\n";
  std::cout << "Perturbation: L_inf epsilon = " << eps << "\n\n";

  bool all_certified_forward = true;
  bool all_certified_backward = true;
  bool all_certified_backward_only = true;

  std::cout << std::fixed << std::setprecision(6);

  for (const auto &x0 : points) {
    const Vector y = network_forward(network, x0);
    const ForwardBoundResult fwd = lirpa_forward_bound(network, x0, eps);
    const BackwardBoundResult bwd = lirpa_backward_bound(network, x0, eps);
    const BackwardBoundResult bwd_only =
        backward_only_verifier.bound(network, x0, eps);
    const int expected = xor_expected_label(x0);

    bool fwd_certified;
    bool bwd_certified;
    bool bwd_only_certified;
    std::string condition;

    if (expected == 1) {
      fwd_certified = (fwd.final_lower.v[0] > 0.5);
      bwd_certified = (bwd.final_lower.v[0] > 0.5);
      bwd_only_certified = (bwd_only.final_lower.v[0] > 0.5);
      condition = "lower bound > 0.5";
    } else {
      fwd_certified = (fwd.final_upper.v[0] < 0.5);
      bwd_certified = (bwd.final_upper.v[0] < 0.5);
      bwd_only_certified = (bwd_only.final_upper.v[0] < 0.5);
      condition = "upper bound < 0.5";
    }

    all_certified_forward = all_certified_forward && fwd_certified;
    all_certified_backward = all_certified_backward && bwd_certified;
    all_certified_backward_only =
        all_certified_backward_only && bwd_only_certified;

    std::cout << "x0=[" << x0.v[0] << ", " << x0.v[1]
              << "], expected=" << expected << ", network_output=" << y.v[0]
              << "\n";
    std::cout << "  forward  bound=[" << fwd.final_lower.v[0] << ", "
              << fwd.final_upper.v[0]
              << "], certified=" << (fwd_certified ? "True" : "False") << " ("
              << condition << ")\n";
    std::cout << "  backward bound=[" << bwd.final_lower.v[0] << ", "
              << bwd.final_upper.v[0]
              << "], certified=" << (bwd_certified ? "True" : "False") << " ("
              << condition << ")\n";
    std::cout << "  backward-only bound=[" << bwd_only.final_lower.v[0] << ", "
              << bwd_only.final_upper.v[0]
              << "], certified=" << (bwd_only_certified ? "True" : "False")
              << " (" << condition << ")\n";
  }

  std::cout << "\n";
  if (all_certified_forward) {
    std::cout << "Forward mode certifies all four XOR corner classifications "
                 "for this epsilon.\n";
  } else {
    std::cout << "Forward mode does not certify at least one XOR corner "
                 "classification for this epsilon.\n";
  }

  if (all_certified_backward) {
    std::cout << "Backward mode certifies all four XOR corner classifications "
                 "for this epsilon.\n";
  } else {
    std::cout << "Backward mode does not certify at least one XOR corner "
                 "classification for this epsilon.\n";
  }

  if (all_certified_backward_only) {
    std::cout << "Backward-only mode certifies all four XOR corner "
                 "classifications for this epsilon.\n";
  } else {
    std::cout << "Backward-only mode does not certify at least one XOR corner "
                 "classification for this epsilon.\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::cout << "Program started...\n";
    double eps = 0.02;
    if (argc >= 2) {
      eps = std::stod(argv[1]);
    }

    std::cout << "Running self-tests...\n";
    self_test_relaxations();
    std::cout << "Self-tests passed.\n";

    run_xor_demo(eps);
    std::cout << "Program finished.\n";
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}

/*
// argc 전달된 인자의 개수 (실행될 프로그램도 인자로 포함되므로 무조건 기본값
1을 가짐)
// argv 인수의 문자열 배열(포인터)

// ex. argv[0] : 프로그램 이름(기본값), argv[1] : 첫번째 추가 인수 ...
int main(int argc, char** argv) {
    try {
        double eps = 0.02; // 엡실론 기본 값
        if (argc >= 2){
            eps = stod(argv[1]); // 엡실론 = 추가 인수 // stod 문자열 -> 실수
변환
        }

        // todo



    }



}
*/