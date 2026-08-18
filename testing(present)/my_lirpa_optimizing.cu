// ============================================================
// my_lirpa_mempool.cu - my_lirpa.cu에 메모리 풀만 적용한 버전
//
// my_lirpa.cu 대비 변경점: 오직 "메모리 풀" 하나만 추가
//   - 모든 함수에서 cudaMalloc/cudaFree 반복 호출 제거
//   - 프로그램 시작 시 GPU 메모리를 미리 할당(풀)해 두고 재사용
//   - GPU 커널, 연산 로직, 알고리즘은 my_lirpa.cu와 100% 동일
//
// 메모리 풀 크기:
//   - Matrix 슬롯 3개 (mat_add, matmul, rowwise_scale 등에서 최대 3개 동시 사용)
//   - Vector 슬롯 6개 (relu_relax에서 최대 6개 동시 사용)
//
// 컴파일: nvcc -Xcompiler "/utf-8" -Xlinker "/STACK:134217728"
//         -arch=sm_89 crown_test_mempool.cu -o crown_test_mempool.exe
// ============================================================

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
#include <nvtx3/nvToolsExt.h>


// Relaxation : 선형 근사를 통한 사잇값 계산

namespace {

constexpr int MAX_LAYERS = 16;
constexpr int MAX_DIM = 512;

enum class ActivationType {
  Relu,
  Sigmoid,
  Linear,
};

// 거의 2MB 의 크기를 가짐 (512 * 512 * 8byte)
struct Matrix {
  int rows = 0;                 // 4 btye
  int cols = 0;                 // 4 byte
  double a[MAX_DIM][MAX_DIM]{}; // 8 * 512 * 512 ==> 2MB
};

struct Vector {
  int n = 0;           // 4byte
  double v[MAX_DIM]{}; // 8byte * 512(2^9)  ==> 4KB
};

struct AffineBound {
  Matrix lower_A;
  Vector lower_c;
  Matrix upper_A;
  Vector upper_c;
};

struct LayerBound {
  int dim = 0;
  Vector alpha_lower;
  Vector beta_lower;
  Vector alpha_upper;
  Vector beta_upper;
};

struct ForwardBoundResult {
  AffineBound final_affine;
  Vector final_lower;
  Vector final_upper;
  int num_layer_bounds = 0;
  LayerBound layer_bounds[MAX_LAYERS]{}; // 순전파 과정에서 얻은 중간 upper, lower bound
};

struct BackwardBoundResult {
  AffineBound final_affine;
  Vector final_lower;
  Vector final_upper;
  int num_layer_bounds = 0;
  LayerBound layer_bounds[MAX_LAYERS]{};
};

struct FullyConnectedNetwork {
  int num_layers = 0;
  int layer_in_dim[MAX_LAYERS]{};
  int layer_out_dim[MAX_LAYERS]{};
  Matrix W[MAX_LAYERS]{};
  Vector b[MAX_LAYERS]{};
  ActivationType act[MAX_LAYERS]{};
};

// ============================================================
// 메모리 풀 (Memory Pool) 
// ============================================================
// 함수들이 사용하는 최대 동시 GPU 버퍼 수:
//   Matrix: 3개 (mat_add: A,B,C / matmul: A,B,C / rowwise_scale: A,out + vec1)
//   Vector: 6개 (relu_relax: lower,upper,alpha_l,beta_l,alpha_u,beta_u)
// affine_min/max는 1 Matrix + 5 Vector이므로 위 범위 안에 들어감

constexpr int NUM_POOL_MAT = 3;
constexpr int NUM_POOL_VEC = 6;
constexpr int NUM_FWD_MAT = 6;
constexpr int NUM_FWD_VEC = 14;
constexpr int NUM_BWD_MAT = 10;
constexpr int NUM_BWD_VEC = 14;

struct GpuPool {
  Matrix* d_mat[NUM_POOL_MAT];        // 2MB * 3 ==> 6MB
  Vector* d_vec[NUM_POOL_VEC];        // 4KB * 6 ==> 24KB
  // 
  Matrix* d_fwd_mat[NUM_FWD_MAT]{};
  Vector* d_fwd_vec[NUM_FWD_VEC]{};
  Vector* d_fwd_alpha_lower[MAX_LAYERS]{};
  Vector* d_fwd_beta_lower[MAX_LAYERS]{};
  Vector* d_fwd_alpha_upper[MAX_LAYERS]{};
  Vector* d_fwd_beta_upper[MAX_LAYERS]{};
  Matrix* d_bwd_mat[NUM_BWD_MAT]{};
  Vector* d_bwd_vec[NUM_BWD_VEC]{};
  // 불변 가중치 메모리 풀에 올려놓기
  Matrix* d_weight[MAX_LAYERS]{};     // 2MB * 16 ==> 32
  Matrix* d_weight_pos[MAX_LAYERS]{}; // 2MB * 16 ==> 32
  Matrix* d_weight_neg[MAX_LAYERS]{}; // 2MB * 16 ==> 32
  Vector* d_bias[MAX_LAYERS]{};
  const FullyConnectedNetwork* cached_network = nullptr;
  bool initialized = false;

  // 1번만 실행되도록
  void init() {
    if (initialized) return;
    for (int i = 0; i < NUM_POOL_MAT; i++) // NUM_POOL_MAT : 3
      cudaMalloc(&d_mat[i], sizeof(Matrix)); // 6MB
    for (int i = 0; i < NUM_POOL_VEC; i++) // NUM_POOL_VEC : 6
      cudaMalloc(&d_vec[i], sizeof(Vector)); // 24kb
    for (int i = 0; i < NUM_FWD_MAT; ++i) // NUM_FWD_MAT : 6
      cudaMalloc(&d_fwd_mat[i], sizeof(Matrix)); // 12MB
    for (int i = 0; i < NUM_FWD_VEC; ++i)   // NUM_FWD_VEC : 14
      cudaMalloc(&d_fwd_vec[i], sizeof(Vector)); // 56kb
    for (int i = 0; i < MAX_LAYERS; ++i) {  // Max_Layers : 16
      cudaMalloc(&d_fwd_alpha_lower[i], sizeof(Vector)); // 4KB * 16 = 64KB
      cudaMalloc(&d_fwd_beta_lower[i], sizeof(Vector)); // 4KB * 16 = 64KB
      cudaMalloc(&d_fwd_alpha_upper[i], sizeof(Vector)); // 4KB * 16 = 64KB
      cudaMalloc(&d_fwd_beta_upper[i], sizeof(Vector)); // 4KB * 16 = 64kb
      cudaMalloc(&d_bias[i], sizeof(Vector)); // 4KB * 16 = 64kb
    }
    for (int i = 0; i < NUM_BWD_MAT; ++i) // NUM_BWD_MAT : 10
      cudaMalloc(&d_bwd_mat[i], sizeof(Matrix)); // 20MB
    for (int i = 0; i < NUM_BWD_VEC; ++i) // NUM_BWD_VEC : 14
      cudaMalloc(&d_bwd_vec[i], sizeof(Vector)); // 56kb
    initialized = true;
  }
  // 6MB + 12MB + 20MB + 24KB + 56KB + 64KB*5 + 56kb ==> 약 38MB + 456kb

  void destroy() {
    if (!initialized) return;
    for (int i = 0; i < NUM_POOL_MAT; i++)
      cudaFree(d_mat[i]);
    for (int i = 0; i < NUM_POOL_VEC; i++)
      cudaFree(d_vec[i]);
    for (int i = 0; i < NUM_FWD_MAT; ++i)
      cudaFree(d_fwd_mat[i]);
    for (int i = 0; i < NUM_FWD_VEC; ++i)
      cudaFree(d_fwd_vec[i]);
    for (int i = 0; i < MAX_LAYERS; ++i) {
      cudaFree(d_fwd_alpha_lower[i]);
      cudaFree(d_fwd_beta_lower[i]);
      cudaFree(d_fwd_alpha_upper[i]);
      cudaFree(d_fwd_beta_upper[i]);
      d_fwd_alpha_lower[i] = nullptr;
      d_fwd_beta_lower[i] = nullptr;
      d_fwd_alpha_upper[i] = nullptr;
      d_fwd_beta_upper[i] = nullptr;
      cudaFree(d_bias[i]);
      d_bias[i] = nullptr;
    }
    for (int i = 0; i < NUM_BWD_MAT; ++i)
      cudaFree(d_bwd_mat[i]);
    for (int i = 0; i < NUM_BWD_VEC; ++i)
      cudaFree(d_bwd_vec[i]);
    // 메모리 해제
    for (int i = 0; i < MAX_LAYERS; ++i) {
      cudaFree(d_weight[i]);
      cudaFree(d_weight_pos[i]);
      cudaFree(d_weight_neg[i]);
      d_weight[i] = nullptr;
      d_weight_pos[i] = nullptr;
      d_weight_neg[i] = nullptr;
    }
    cached_network = nullptr;
    initialized = false;
  }
};

static GpuPool g_pool;

void ensure_pool() {
  if (!g_pool.initialized) g_pool.init();
}

void gpu_pool_cleanup() {
  g_pool.destroy();
}

// ============================================================
// 기본 유틸리티
// ============================================================

__host__ __device__ inline double pos(double x) { return x > 0.0 ? x : 0.0; }
__host__ __device__ inline double neg(double x) { return x < 0.0 ? x : 0.0; }
__host__ __device__ inline double relu(double x) { return x > 0.0 ? x : 0.0; }

__host__ __device__ inline double sigmoid(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + exp(-x));
  }
  const double ex = exp(x);
  return ex / (1.0 + ex);
}

__host__ __device__ inline double sigmoid_prime(double x) {
  const double s = sigmoid(x);
  return s * (1.0 - s);
}

void require(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument(msg);
  }
}

// ============================================================
// make 함수들 - cpu + 풀 사용으로 변경
// ============================================================

Matrix make_zero_matrix(int rows, int cols) {
  require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM,
          "Matrix shape out of bounds.");

  Matrix out{};       // CPU에서 0 초기화
  out.rows = rows;
  out.cols = cols;
  return out;
}

Vector make_zero_vector(int n) {
  require(n >= 0 && n <= MAX_DIM, "Vector length out of bounds.");

  Vector out{};       // CPU에서 0 초기화
  out.n = n;
  return out;
}

// --- CPU 원본 함수 (성능 비교 및 백업용) ---
Matrix make_eye_cpu(int n) {
  Matrix out = make_zero_matrix(n, n);
  for (int i = 0; i < n; ++i) {
    out.a[i][i] = 1.0;
  }
  return out;
}

__global__ void make_eye_gpu(Matrix *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out->a[tid][tid] = 1.0;
  }
}

// 단위행렬 만들기
Matrix make_eye(int n) {
  ensure_pool();
  Matrix out = make_zero_matrix(n, n);

  cudaMemcpy(g_pool.d_mat[0], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid == 0) blocksPerGrid = 1;

  make_eye_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], n);

  cudaMemcpy(&out, g_pool.d_mat[0], sizeof(Matrix), cudaMemcpyDeviceToHost);
  return out;
}

// ============================================================
// GPU 커널 함수들 (my_lirpa.cu와 동일)
// ============================================================

// 행렬 합
__global__ void mat_add_gpu(const Matrix *A, const Matrix *B, Matrix *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;
  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    C->a[r][c] = A->a[r][c] + B->a[r][c];
  }
}

// 풀 사용 버전: cudaMalloc/cudaFree 제거
Matrix mat_add(const Matrix &A, const Matrix &B) {
  require(A.rows == B.rows && A.cols == B.cols, "mat_add shape mismatch.");
  ensure_pool();
  Matrix C = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_mat[1], &B, sizeof(Matrix), cudaMemcpyHostToDevice);
  // Output is fully overwritten by mat_add_gpu; no H2D initialization needed.
  // cudaMemcpy(g_pool.d_mat[2], &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  mat_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1], g_pool.d_mat[2]);

  cudaMemcpy(&C, g_pool.d_mat[2], sizeof(Matrix), cudaMemcpyDeviceToHost);
  C.rows = A.rows;
  C.cols = A.cols;
  return C;
}

// 벡터 합
__global__ void vec_add_gpu(const Vector *A, const Vector *B, Vector *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < A->n) {
    C->v[tid] = A->v[tid] + B->v[tid];
  }
}

Vector vec_add(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_add shape mismatch.");
  ensure_pool();
  Vector c = make_zero_vector(a.n);

  cudaMemcpy(g_pool.d_vec[0], &a, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &b, sizeof(Vector), cudaMemcpyHostToDevice);
  // 굳이 c를 GPU에서 복사하지 말고 (어짜피 의미가 없는 0벡터니까)
  // cudaMemcpy(g_pool.d_vec[2], &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  vec_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2]);

  cudaMemcpy(&c, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
  c.n = a.n;
  return c;
}

// 벡터 빼기
__global__ void vec_sub_gpu(const Vector *A, const Vector *B, Vector *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < A->n) {
    C->v[tid] = A->v[tid] - B->v[tid];
  }
}

Vector vec_sub(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_sub shape mismatch.");
  ensure_pool();
  Vector c = make_zero_vector(a.n);

  cudaMemcpy(g_pool.d_vec[0], &a, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &b, sizeof(Vector), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_vec[2], &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  vec_sub_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2]);

  cudaMemcpy(&c, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
  c.n = a.n; // 벡터 열의 개수 업데이트 (shape)
  return c;
}

// 행렬 곱셈
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
  ensure_pool();
  Matrix C = make_zero_matrix(A.rows, B.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_mat[1], &B, sizeof(Matrix), cudaMemcpyHostToDevice);
  // 어짜피 Matrix C의 행렬은 A * B의 행렬로 덮어씌워질 예정이므로 복사가 필요 없음
  // cudaMemcpy(g_pool.d_mat[2], &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * B.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1], g_pool.d_mat[2]);

  cudaMemcpy(&C, g_pool.d_mat[2], sizeof(Matrix), cudaMemcpyDeviceToHost);
  C.rows = A.rows;
  C.cols = B.cols;
  return C;
}

// 가중치가 GPU에 미리 올라와 있는 상태의 행렬 곱셈 함수 정의

// A is still a host-side intermediate, but B is an immutable matrix already
// resident on the GPU.  This removes one full Matrix H2D transfer per call.
Matrix matmul_device_rhs(const Matrix &A, const Matrix *d_B, int b_rows,
                         int b_cols) {
  require(A.cols == b_rows, "matmul_device_rhs shape mismatch.");
  ensure_pool();
  Matrix C = make_zero_matrix(A.rows, b_cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);

  const int total_elements = A.rows * b_cols;
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
  matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], d_B,
                                                   g_pool.d_mat[2]);

  cudaMemcpy(&C, g_pool.d_mat[2], sizeof(Matrix), cudaMemcpyDeviceToHost);
  C.rows = A.rows;
  C.cols = b_cols;
  return C;
}

// Device-resident left operand variant for W * A in the forward bound.
Matrix matmul_device_lhs(const Matrix *d_A, int a_rows, int a_cols,
                         const Matrix &B) {
  require(a_cols == B.rows, "matmul_device_lhs shape mismatch.");
  ensure_pool();
  Matrix C = make_zero_matrix(a_rows, B.cols);

  cudaMemcpy(g_pool.d_mat[1], &B, sizeof(Matrix), cudaMemcpyHostToDevice);

  const int total_elements = a_rows * B.cols;
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
  matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, g_pool.d_mat[1],
                                                   g_pool.d_mat[2]);

  cudaMemcpy(&C, g_pool.d_mat[2], sizeof(Matrix), cudaMemcpyDeviceToHost);
  C.rows = a_rows;
  C.cols = B.cols;
  return C;
}

// 행렬-벡터 곱 (Matrix 1개 + Vector 2개 = 풀 안에 들어감)
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

// Device-resident bias variant used by the normal network forward pass.
__global__ void matvec_bias_gpu(const Matrix *A, const Vector *x,
                                const Vector *bias, Vector *y) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) y->n = A->rows;
  if (tid < A->rows) {
    double sum = bias->v[tid];
    for (int j = 0; j < A->cols; ++j) {
      sum += A->a[tid][j] * x->v[j];
    }
    y->v[tid] = sum;
  }
}

Vector matvec_bias_device_matrix(const Matrix *d_A, int rows, int cols,
                                 const Vector *d_bias, const Vector &x) {
  require(cols == x.n, "matvec_bias_device_matrix shape mismatch.");
  ensure_pool();
  Vector y = make_zero_vector(rows);

  cudaMemcpy(g_pool.d_vec[0], &x, sizeof(Vector), cudaMemcpyHostToDevice);

  const int blocksPerGrid = (rows + 255) / 256;
  matvec_bias_gpu<<<blocksPerGrid, 256>>>(d_A, g_pool.d_vec[0], d_bias,
                                          g_pool.d_vec[1]);

  cudaMemcpy(&y, g_pool.d_vec[1], sizeof(Vector), cudaMemcpyDeviceToHost);
  y.n = rows;
  return y;
}

Vector matvec(const Matrix &A, const Vector &x) {
  require(A.cols == x.n, "matvec shape mismatch.");
  ensure_pool();
  Vector y = make_zero_vector(A.rows);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[0], &x, sizeof(Vector), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_vec[1], &y, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  matvec_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0], g_pool.d_vec[1]);

  cudaMemcpy(&y, g_pool.d_vec[1], sizeof(Vector), cudaMemcpyDeviceToHost);
  y.n = A.rows;
  return y;
}

// 동일하게 가중치가 미리 올라와 있는 행렬 * 벡터 곱
// Device-resident matrix variant used for immutable network weights.
Vector matvec_device_matrix(const Matrix *d_A, int rows, int cols,
                            const Vector &x) {
  require(cols == x.n, "matvec_device_matrix shape mismatch.");
  ensure_pool();
  Vector y = make_zero_vector(rows);

  cudaMemcpy(g_pool.d_vec[0], &x, sizeof(Vector), cudaMemcpyHostToDevice);

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
  matvec_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_A, g_pool.d_vec[0],
                                                   g_pool.d_vec[1]);

  cudaMemcpy(&y, g_pool.d_vec[1], sizeof(Vector), cudaMemcpyDeviceToHost);
  y.n = rows;
  return y;
}

// positive_part (Matrix 2개 사용)
__global__ void positive_part_gpu(const Matrix *A, Matrix *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;
  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    out->a[r][c] = pos(A->a[r][c]);
  }
}

// 행렬 양수부분만 남기기
Matrix positive_part(const Matrix &A) {
  ensure_pool();
  Matrix out = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_mat[1], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  positive_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1]);

  cudaMemcpy(&out, g_pool.d_mat[1], sizeof(Matrix), cudaMemcpyDeviceToHost);
  out.rows = A.rows;
  out.cols = A.cols;
  return out;
}

// Vector 양수부분만 남기기 (코드 상 분석으로는 총 4번 호출하는데 cpu로 해도 될지도? (O(n)이니까))
// 혹시 모르니까 주석 기록은 남겨놓음(필요시 바꿀 수 있게) <== 이것도 Gpool에 올려놓기?
Vector positive_part(const Vector& x) {
    Vector out = make_zero_vector(x.n);
    for (int i = 0; i < x.n; ++i) {
        out.v[i] = pos(x.v[i]);
    }
    return out;
}

// negative_part (Matrix 2개 사용)
__global__ void negative_part_gpu(const Matrix *A, Matrix *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;
  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    out->a[r][c] = neg(A->a[r][c]);
  }
}

// 행렬 음수부분만 남기기
Matrix negative_part(const Matrix &A) {
  ensure_pool();
  Matrix out = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_mat[1], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  negative_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1]);

  cudaMemcpy(&out, g_pool.d_mat[1], sizeof(Matrix), cudaMemcpyDeviceToHost);
  out.rows = A.rows;
  out.cols = A.cols;
  return out;
}

// Vector 음수부분만 남기기 (총 4번 호출) (코드 상 분석으로는 총 4번 호출하는데 cpu로 해도 될지도? (O(n)이니까))
// 혹시 모르니까 주석 기록은 남겨놓음(필요시 바꿀 수 있게) <== 이것도 Gpool에 올려놓기? , 고정값인지는 아직 모름
Vector negative_part(const Vector& x) {
    Vector out = make_zero_vector(x.n);
    for (int i = 0; i < x.n; ++i) {
        out.v[i] = neg(x.v[i]);
    }
    return out;
}

// Upload immutable network weights once.  The positive/negative decompositions
// are also computed once and reused by every forward-bound operation.
void prepare_network_on_gpu(const FullyConnectedNetwork &net) {
  ensure_pool();

  // 한번만 실행되도록
  if (g_pool.cached_network == &net) return;

  // GPU 메모리 할당 및 주소 저장
  for (int l = 0; l < net.num_layers; ++l) { // 가중치 레이어 3개
    if (g_pool.d_weight[l] == nullptr) {        
      cudaMalloc(&g_pool.d_weight[l], sizeof(Matrix)); // 2MB * 3  
      cudaMalloc(&g_pool.d_weight_pos[l], sizeof(Matrix)); // 2MB * 3
      cudaMalloc(&g_pool.d_weight_neg[l], sizeof(Matrix)); // 2MB * 3
    }
    // 6MB + 6MB + 6MB => 18MB

    // Seed pos/neg buffers with metadata (rows/cols); their value arrays are
    // overwritten by the kernels below.
    // 가중치 GPU 메모리에 복사
    //nvtxRangePushA("memcpy W,B");
    cudaMemcpy(g_pool.d_weight[l], &net.W[l], sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pool.d_weight_pos[l], &net.W[l], sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pool.d_weight_neg[l], &net.W[l], sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(g_pool.d_bias[l], &net.b[l], sizeof(Vector), cudaMemcpyHostToDevice);
    //nvtxRangePop();

    //미리 가중치 양수/음수 부분만 남겨서 메모리 풀에 저장
    //나중에 forward 같은 데에서 계속 이걸 계산할 필요가 사라짐
    const int total_elements = net.W[l].rows * net.W[l].cols;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    positive_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        g_pool.d_weight[l], g_pool.d_weight_pos[l]);
    negative_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        g_pool.d_weight[l], g_pool.d_weight_neg[l]);
  }
  g_pool.cached_network = &net;
}

// rowwise_scale (Matrix 2개 + Vector 1개 사용)
__global__ void rowwise_scale_gpu(const Matrix *A, const Vector *s, Matrix *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int total_elements = A->rows * A->cols;
  if (tid < total_elements) {
    int r = tid / A->cols;
    int c = tid % A->cols;
    out->a[r][c] = A->a[r][c] * s->v[r];
  }
}

// 행렬의 i번 째 행(원소들과) 벡터의 i번째 원소를 곱하기
Matrix rowwise_scale(const Matrix &A, const Vector &s) {
  require(A.rows == s.n, "rowwise_scale shape mismatch.");
  ensure_pool();
  Matrix out = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[0], &s, sizeof(Vector), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_mat[1], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  rowwise_scale_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0], g_pool.d_mat[1]);

  cudaMemcpy(&out, g_pool.d_mat[1], sizeof(Matrix), cudaMemcpyDeviceToHost);
  out.rows = A.rows;
  out.cols = A.cols;
  return out;
}

// backward_only_iteration에 필요한 함수 
Matrix colwise_scale(const Matrix& A, const Vector& s) {
    require(A.cols == s.n, "colwise_scale shape mismatch.");
    Matrix out = make_zero_matrix(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            out.a[i][j] = A.a[i][j] * s.v[j];
        }
    }
    return out;
}

// elemwise_mul (Vector 3개 사용) 원소 개수가 같은 벡터의 같은 인덱스의 원소끼리 곱하기
__global__ void elemwise_mul_gpu(const Vector *A, const Vector *B, Vector *C) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < A->n) {
    C->v[tid] = A->v[tid] * B->v[tid];
  }
}

Vector elemwise_mul(const Vector &a, const Vector &b) {
  require(a.n == b.n, "elemwise_mul shape mismatch.");
  ensure_pool();
  Vector c = make_zero_vector(a.n);

  cudaMemcpy(g_pool.d_vec[0], &a, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &b, sizeof(Vector), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_vec[2], &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  elemwise_mul_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2]);

  cudaMemcpy(&c, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
  c.n = a.n;
  return c;
}

// CPU 원본 (비교용)
Vector make_eps_vec_cpu(int n, double eps) {
  Vector out = make_zero_vector(n);
  for (int i = 0; i < n; ++i) {
    out.v[i] = eps;
  }
  return out;
}

// 엡실론 벡터 만들기 (Vector 1개 사용)
__global__ void make_eps_vec_gpu(Vector *out, int n, double eps) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out->v[tid] = eps;
  }
}

Vector make_eps_vec(int n, double eps) {
  ensure_pool();
  Vector out = make_zero_vector(n);

  // cudaMemcpy(g_pool.d_vec[0], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid == 0) blocksPerGrid = 1;

  make_eps_vec_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], n, eps);

  cudaMemcpy(&out, g_pool.d_vec[0], sizeof(Vector), cudaMemcpyDeviceToHost);
  out.n = n;
  return out;
}

// affine_min (Matrix 1개 + Vector 5개 사용)
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
  ensure_pool();

  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);
  Vector out = make_zero_vector(A.rows);

  // Matrix 1개 + Vector 5개 사용 (풀 범위 내)
  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[0], &c, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &xl, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[2], &xu, sizeof(Vector), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_vec[3], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  affine_min_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0],
                                                     g_pool.d_vec[1], g_pool.d_vec[2],
                                                     g_pool.d_vec[3]);

  cudaMemcpy(&out, g_pool.d_vec[3], sizeof(Vector), cudaMemcpyDeviceToHost);
  out.n = A.rows;
  return out;
}

// affine_max (Matrix 1개 + Vector 5개 사용)
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
  ensure_pool();

  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);
  Vector out = make_zero_vector(A.rows);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[0], &c, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &xl, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[2], &xu, sizeof(Vector), cudaMemcpyHostToDevice);
  // cudaMemcpy(g_pool.d_vec[3], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  affine_max_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0],
                                                     g_pool.d_vec[1], g_pool.d_vec[2],
                                                     g_pool.d_vec[3]);

  cudaMemcpy(&out, g_pool.d_vec[3], sizeof(Vector), cudaMemcpyDeviceToHost);
  out.n = A.rows;
  return out;
}

// relu_relax (Vector 6개 사용 - 풀 최대)
__global__ void relu_relax_gpu(const Vector *lower, const Vector *upper,
                               Vector *alpha_l, Vector *beta_l, Vector *alpha_u,
                               Vector *beta_u) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < lower->n) {
    double l = lower->v[tid];
    double u = upper->v[tid];
    assert(l <= u);

    if (l >= 0.0) {
      alpha_l->v[tid] = 1.0;
      alpha_u->v[tid] = 1.0;
    } else if (u <= 0.0) {
      // keep zeros
    } else {
      double denom = u - l;
      alpha_u->v[tid] = u / denom;
      beta_u->v[tid] = -u * l / denom;

      bool use_identity_lower = fabs(l) < fabs(u);
      alpha_l->v[tid] = use_identity_lower ? 1.0 : 0.0;
      beta_l->v[tid] = 0.0;
    }
  }
}

// GPU-resident forward-bound helpers.  Kernels write only value arrays, so
// keep the small shape metadata valid on every reusable workspace buffer.
inline void set_matrix_shape(Matrix *d_matrix, int rows, int cols) {
  const int shape[2] = {rows, cols};
  cudaMemcpy(d_matrix, shape, sizeof(shape), cudaMemcpyHostToDevice);
}

inline void set_vector_size(Vector *d_vector, int n) {
  cudaMemcpy(d_vector, &n, sizeof(n), cudaMemcpyHostToDevice);
}

__global__ void matmul_pair_add_gpu(const Matrix *A1, const Matrix *B1,
                                    const Matrix *A2, const Matrix *B2,
                                    Matrix *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = A1->rows * B1->cols;
  if (tid == 0) {
    out->rows = A1->rows;
    out->cols = B1->cols;
  }
  if (tid < total) {
    const int r = tid / B1->cols;
    const int c = tid % B1->cols;
    double sum = 0.0;
    for (int k = 0; k < A1->cols; ++k) {
      sum += A1->a[r][k] * B1->a[k][c]
           + A2->a[r][k] * B2->a[k][c];
    }
    out->a[r][c] = sum;
  }
}

__global__ void matvec_pair_bias_gpu(const Matrix *A1, const Vector *x1,
                                     const Matrix *A2, const Vector *x2,
                                     const Vector *bias, Vector *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) out->n = A1->rows;
  if (tid < A1->rows) {
    double sum = bias->v[tid];
    for (int j = 0; j < A1->cols; ++j) {
      sum += A1->a[tid][j] * x1->v[j]
           + A2->a[tid][j] * x2->v[j];
    }
    out->v[tid] = sum;
  }
}

__global__ void elemwise_affine_fused_gpu(const Vector *alpha,
                                           const Vector *value,
                                           const Vector *beta, Vector *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < alpha->n) {
    out->v[tid] = alpha->v[tid] * value->v[tid] + beta->v[tid];
  }
}

__global__ void elemwise_affine_pair_gpu(
    const Vector *lower_alpha, const Vector *lower_value,
    const Vector *lower_beta, const Vector *upper_alpha,
    const Vector *upper_value, const Vector *upper_beta,
    Vector *lower_out, Vector *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    lower_out->n = lower_alpha->n;
    upper_out->n = lower_alpha->n;
  }
  if (tid < lower_alpha->n) {
    lower_out->v[tid] = lower_alpha->v[tid] * lower_value->v[tid]
                      + lower_beta->v[tid];
    upper_out->v[tid] = upper_alpha->v[tid] * upper_value->v[tid]
                      + upper_beta->v[tid];
  }
}

__global__ void rowwise_scale_pair_gpu(
    const Matrix *lower_in, const Vector *lower_scale,
    const Matrix *upper_in, const Vector *upper_scale,
    Matrix *lower_out, Matrix *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = lower_in->rows * lower_in->cols;
  if (tid == 0) {
    lower_out->rows = lower_in->rows;
    lower_out->cols = lower_in->cols;
    upper_out->rows = upper_in->rows;
    upper_out->cols = upper_in->cols;
  }
  if (tid < total) {
    const int r = tid / lower_in->cols;
    const int c = tid % lower_in->cols;
    lower_out->a[r][c] = lower_in->a[r][c] * lower_scale->v[r];
    upper_out->a[r][c] = upper_in->a[r][c] * upper_scale->v[r];
  }
}

__global__ void affine_minmax_pair_gpu(
    const Matrix *lower_A, const Vector *lower_c,
    const Matrix *upper_A, const Vector *upper_c,
    const Vector *xl, const Vector *xu,
    Vector *lower_out, Vector *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    lower_out->n = lower_A->rows;
    upper_out->n = upper_A->rows;
  }
  if (tid < lower_A->rows) {
    double lower_sum = lower_c->v[tid];
    double upper_sum = upper_c->v[tid];
    for (int j = 0; j < lower_A->cols; ++j) {
      const double lower_value = lower_A->a[tid][j];
      const double upper_value = upper_A->a[tid][j];
      lower_sum += pos(lower_value) * xl->v[j]
                 + neg(lower_value) * xu->v[j];
      upper_sum += pos(upper_value) * xu->v[j]
                 + neg(upper_value) * xl->v[j];
    }
    lower_out->v[tid] = lower_sum;
    upper_out->v[tid] = upper_sum;
  }
}

__global__ void cache_relaxation_layer_gpu(
    const Vector *alpha_l, const Vector *beta_l,
    const Vector *alpha_u, const Vector *beta_u,
    Vector *cached_alpha_l, Vector *cached_beta_l,
    Vector *cached_alpha_u, Vector *cached_beta_u) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    cached_alpha_l->n = alpha_l->n;
    cached_beta_l->n = beta_l->n;
    cached_alpha_u->n = alpha_u->n;
    cached_beta_u->n = beta_u->n;
  }
  if (tid < alpha_l->n) {
    cached_alpha_l->v[tid] = alpha_l->v[tid];
    cached_beta_l->v[tid] = beta_l->v[tid];
    cached_alpha_u->v[tid] = alpha_u->v[tid];
    cached_beta_u->v[tid] = beta_u->v[tid];
  }
}

__global__ void relu_relax_full_gpu(const Vector *lower, const Vector *upper,
                                    Vector *alpha_l, Vector *beta_l,
                                    Vector *alpha_u, Vector *beta_u) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    alpha_l->n = lower->n;
    beta_l->n = lower->n;
    alpha_u->n = lower->n;
    beta_u->n = lower->n;
  }
  if (tid >= lower->n) return;
  const double l = lower->v[tid];
  const double u = upper->v[tid];
  if (l >= 0.0) {
    alpha_l->v[tid] = 1.0; beta_l->v[tid] = 0.0;
    alpha_u->v[tid] = 1.0; beta_u->v[tid] = 0.0;
  } else if (u <= 0.0) {
    alpha_l->v[tid] = 0.0; beta_l->v[tid] = 0.0;
    alpha_u->v[tid] = 0.0; beta_u->v[tid] = 0.0;
  } else {
    const double slope = u / (u - l);
    alpha_u->v[tid] = slope; beta_u->v[tid] = -u * l / (u - l);
    alpha_l->v[tid] = fabs(l) < fabs(u) ? 1.0 : 0.0;
    beta_l->v[tid] = 0.0;
  }
}

__global__ void linear_relax_full_gpu(Vector *alpha_l, Vector *beta_l,
                                      Vector *alpha_u, Vector *beta_u, int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    alpha_l->n = n;
    beta_l->n = n;
    alpha_u->n = n;
    beta_u->n = n;
  }
  if (tid < n) {
    alpha_l->v[tid] = 1.0; beta_l->v[tid] = 0.0;
    alpha_u->v[tid] = 1.0; beta_u->v[tid] = 0.0;
  }
}

void relu_relax(const Vector &lower, const Vector &upper, Vector &alpha_l,
                Vector &beta_l, Vector &alpha_u, Vector &beta_u) {
  require(lower.n == upper.n, "relu_relax shape mismatch.");
  ensure_pool();

  alpha_l = make_zero_vector(lower.n);
  beta_l = make_zero_vector(lower.n);
  alpha_u = make_zero_vector(lower.n);
  beta_u = make_zero_vector(lower.n);

  // Vector 슬롯 6개 전부 사용
  cudaMemcpy(g_pool.d_vec[0], &lower, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &upper, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[2], &alpha_l, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[3], &beta_l, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[4], &alpha_u, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[5], &beta_u, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (lower.n + threadsPerBlock - 1) / threadsPerBlock;

  relu_relax_gpu<<<blocksPerGrid, threadsPerBlock>>>(
      g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2], g_pool.d_vec[3],
      g_pool.d_vec[4], g_pool.d_vec[5]);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in relu_relax_gpu launch: "
              << cudaGetErrorString(err) << "\n";
  }

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA error during relu_relax_gpu execution: "
              << cudaGetErrorString(err) << "\n";
    throw std::runtime_error("relu_relax_gpu 실행 중 에러 발생");
  }

  cudaMemcpy(&alpha_l, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(&beta_l, g_pool.d_vec[3], sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(&alpha_u, g_pool.d_vec[4], sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(&beta_u, g_pool.d_vec[5], sizeof(Vector), cudaMemcpyDeviceToHost);
}

// ============================================================
// sigmoid_relax (CPU - my_lirpa.cu와 동일, GPU 포팅 불필요)
// ============================================================

double bisect_root(double lo, double hi,
                   const std::function<double(double)> &fn, int max_iter = 80,
                   double tol = 1e-12) {
  double flo = fn(lo);
  double fhi = fn(hi);

  if (std::abs(flo) < tol) return lo;
  if (std::abs(fhi) < tol) return hi;

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
      if (cur < best_abs) { best_abs = cur; best = i; }
    }
    bool found = false;
    for (int i = 0; i < GRID - 1; ++i) {
      if (vals[i] == 0.0 || vals[i] * vals[i + 1] <= 0.0) {
        lo = xs[i]; hi = xs[i + 1]; flo = vals[i]; found = true; break;
      }
    }
    if (!found) return xs[best];
  }

  for (int it = 0; it < max_iter; ++it) {
    const double mid = 0.5 * (lo + hi);
    const double fmid = fn(mid);
    if (std::abs(fmid) < tol || std::abs(hi - lo) < tol) return mid;
    if (flo * fmid <= 0.0) { hi = mid; }
    else { lo = mid; flo = fmid; }
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
      alpha_l.v[i] = slope; beta_l.v[i] = intercept;
      alpha_u.v[i] = slope; beta_u.v[i] = intercept;
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
      lower_violation = std::max(lower_violation, alpha_l.v[i] * x + beta_l.v[i] - y);
      upper_violation = std::max(upper_violation, y - (alpha_u.v[i] * x + beta_u.v[i]));
    }
    if (lower_violation > 1e-10) { beta_l.v[i] -= lower_violation + 1e-10; }
    if (upper_violation > 1e-10) { beta_u.v[i] += upper_violation + 1e-10; }
  }
}

// ============================================================
// 신경망 관련 함수 (my_lirpa.cu와 동일)
// ============================================================

ActivationType parse_activation(const std::string &name) {
  std::string lower = name;
  for (char &ch : lower) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (lower == "relu") return ActivationType::Relu;
  if (lower == "sigmoid") return ActivationType::Sigmoid;
  if (lower == "linear") return ActivationType::Linear;
  throw std::invalid_argument("Unsupported activation: " + name);
}

FullyConnectedNetwork make_network(int num_layers, const int *layer_in_dim, const int *layer_out_dim,
             const double W_data[MAX_LAYERS][MAX_DIM][MAX_DIM],
             const double b_data[MAX_LAYERS][MAX_DIM],
             const std::string *activations) {
  require(num_layers >= 1 && num_layers <= MAX_LAYERS, "num_layers out of bounds.");
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

__global__ void apply_activation_gpu(ActivationType act, Vector *A){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < A->n){
    if(act == ActivationType::Relu){
      A->v[tid] = relu(A->v[tid]);
    }
    else if(act == ActivationType::Sigmoid){
      A->v[tid] = sigmoid(A->v[tid]);
    }
    else if(act == ActivationType::Linear){
      //unchanged
    }
  }
}

Vector apply_activation(const Vector &s, const ActivationType act){
  ensure_pool();
  Vector out = s;

  cudaMemcpy(g_pool.d_vec[0], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (out.n + threadsPerBlock - 1) / threadsPerBlock;

  apply_activation_gpu<<<blocksPerGrid, threadsPerBlock>>>(act, g_pool.d_vec[0]);

  cudaMemcpy(&out, g_pool.d_vec[0], sizeof(Vector), cudaMemcpyDeviceToHost);
  return out;
}

// 신경망 순전파 (추론)
// 인자로 신경망 , 입력값(256개)
Vector network_forward(const FullyConnectedNetwork &net, const Vector &x) {
  require(x.n == network_input_dim(net),
          "network_forward input dimension mismatch.");
  prepare_network_on_gpu(net);
  
  // 1. 입력 데이터를 처음에 단 한 번만 GPU로 복사 (H2D)
  cudaMemcpy(g_pool.d_vec[0], &x, sizeof(Vector), cudaMemcpyHostToDevice);
  
  // 사용할 입력, 출력 벡터 포인터 선언
  Vector *d_in = g_pool.d_vec[0];
  Vector *d_out = g_pool.d_vec[1];

  for (int l = 0; l < net.num_layers; ++l) {
    const int rows = net.W[l].rows;
    const int blocksPerGrid = (rows + 255) / 256;
    
    // 2. GPU 내부에서 행렬곱 + Bias 덧셈 실행
    matvec_bias_gpu<<<blocksPerGrid, 256>>>(g_pool.d_weight[l], d_in, g_pool.d_bias[l], d_out);
    
    // 3. GPU 내부에서 활성화 함수 적용
    // 여기서 CPU 메모리에 있는 net.act[l]를 사용했는데, CUDA 커널 인자 방식을 이해해야함
    apply_activation_gpu<<<blocksPerGrid, 256>>>(net.act[l], d_out);
    
    // 4. 다음 레이어의 입력을 현재 출력으로 스왑
    std::swap(d_in, d_out);
  }
  
  // 5. 모든 레이어를 통과한 최종 결과만 단 한 번 CPU로 복사 (D2H)
  // (마지막 루프에서 스왑되었으므로 최종 결과는 d_in에 있음)
  Vector f = make_zero_vector(network_output_dim(net));
  cudaMemcpy(&f, d_in, sizeof(Vector), cudaMemcpyDeviceToHost);
  
  return f;
}

// ============================================================
// CROWN 알고리즘 (my_lirpa.cu와 동일한 로직)
// ============================================================

// materialze_host_result : 호스트(cpu)로 전달(memcpy)해줄지 결정 , true면 전달 false면 전달x
ForwardBoundResult lirpa_forward_bound_impl(const FullyConnectedNetwork &net,
                                            const Vector &x0, double eps,
                                            bool materialize_host_results) {
  require(x0.n == network_input_dim(net),
          "lirpa_forward_bound input dimension mismatch.");
  prepare_network_on_gpu(net);
  ForwardBoundResult out;
  out.num_layer_bounds = net.num_layers;
  // =====================================================================
  // 핑퐁(Ping-Pong)을 위한 고정 GPU 버퍼 맵핑
  // 원본의 current.lower_A, pre.lower_A, post.lower_A 등을 
  // 매번 새로 만들지 않고 아래의 6개 행렬, 14개 벡터 버퍼 안에서 돌려막기 합니다.
  // =====================================================================
  Matrix *const lower_A = g_pool.d_fwd_mat[0];     // 원본: current.lower_A (동시에 post 역할도 겸함)
  Matrix *const upper_A = g_pool.d_fwd_mat[1];     // 원본: current.upper_A 
  Matrix *const pre_lower_A = g_pool.d_fwd_mat[2]; // 원본: pre.lower_A
  Matrix *const pre_upper_A = g_pool.d_fwd_mat[3]; // 원본: pre.upper_A
  Matrix *const tmp_A = g_pool.d_fwd_mat[4];       // 계산용 임시 도마 1
  Matrix *const tmp_B = g_pool.d_fwd_mat[5];       // 계산용 임시 도마 2
  Vector *const lower_c = g_pool.d_fwd_vec[0];     // 원본: current.lower_c
  Vector *const upper_c = g_pool.d_fwd_vec[1];     // 원본: current.upper_c
  Vector *const pre_lower_c = g_pool.d_fwd_vec[2]; // 원본: pre.lower_c
  Vector *const pre_upper_c = g_pool.d_fwd_vec[3]; // 원본: pre.upper_c
  Vector *const pre_lower = g_pool.d_fwd_vec[4];   // 원본: pre_lower (입력 최솟값)
  Vector *const pre_upper = g_pool.d_fwd_vec[5];   // 원본: pre_upper (입력 최댓값)
  Vector *const alpha_l = g_pool.d_fwd_vec[6];     // 원본: alpha_l
  Vector *const beta_l = g_pool.d_fwd_vec[7];      // 원본: beta_l
  Vector *const alpha_u = g_pool.d_fwd_vec[8];     // 원본: alpha_u
  Vector *const beta_u = g_pool.d_fwd_vec[9];      // 원본: beta_u
  Vector *const d_xl = g_pool.d_fwd_vec[12];       // 입력 박스 최솟값 (x0 - eps)
  Vector *const d_xu = g_pool.d_fwd_vec[13];       // 입력 박스 최댓값 (x0 + eps)


  const int in_dim = network_input_dim(net);
  cudaMemset(lower_A, 0, sizeof(Matrix));
  cudaMemset(upper_A, 0, sizeof(Matrix));
  set_matrix_shape(lower_A, in_dim, in_dim);
  set_matrix_shape(upper_A, in_dim, in_dim);
  make_eye_gpu<<<(in_dim + 255) / 256, 256>>>(lower_A, in_dim);
  make_eye_gpu<<<(in_dim + 255) / 256, 256>>>(upper_A, in_dim);
  cudaMemset(lower_c, 0, sizeof(Vector)); set_vector_size(lower_c, in_dim);
  cudaMemset(upper_c, 0, sizeof(Vector)); set_vector_size(upper_c, in_dim);
  Vector xl = x0, xu = x0;
  for (int i = 0; i < in_dim; ++i) { xl.v[i] -= eps; xu.v[i] += eps; }
  cudaMemcpy(d_xl, &xl, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xu, &xu, sizeof(Vector), cudaMemcpyHostToDevice);

  for (int l = 0; l < net.num_layers; ++l) {
    const Matrix *d_W_pos = g_pool.d_weight_pos[l];
    const Matrix *d_W_neg = g_pool.d_weight_neg[l];
    const int weight_rows = net.W[l].rows;
    const int matrix_elements = weight_rows * in_dim;
    const int matrix_blocks = (matrix_elements + 255) / 256;
    // ---------------------------------------------------------
    // [Pre-activation 계산] 원본: pre.lower_A = W_pos*lower_A + W_neg*upper_A
    // ---------------------------------------------------------
    matmul_pair_add_gpu<<<matrix_blocks, 256>>>(
        d_W_pos, lower_A, d_W_neg, upper_A, pre_lower_A);
    
    // 원본: pre.upper_A = W_pos*upper_A + W_neg*lower_A
    matmul_pair_add_gpu<<<matrix_blocks, 256>>>(
        d_W_pos, upper_A, d_W_neg, lower_A, pre_upper_A);

    // ---------------------------------------------------------
    // [편향(Bias) 계산] 원본: pre.lower_c = W_pos*lower_c + W_neg*upper_c + b
    // ---------------------------------------------------------
    const int vector_blocks = (weight_rows + 255) / 256;
    const Vector *d_bias = g_pool.d_bias[l];
    matvec_pair_bias_gpu<<<vector_blocks, 256>>>(
        d_W_pos, lower_c, d_W_neg, upper_c, d_bias, pre_lower_c);
    // pre_lower_c is reused as the next kernel's input, so its header must
    // be valid before that kernel reads pre_lower_c->n.
    
    // CPU에 있는 net.b[l]만 어쩔 수 없이 아주 잠깐 복사해옴 (크기가 작아 부담 적음)
    // pre.lower_c = pre.lower_c + b (덮어쓰기 In-place 연산!)
    
    // 원본: pre.upper_c = W_pos*upper_c + W_neg*lower_c + b
    matvec_pair_bias_gpu<<<vector_blocks, 256>>>(
        d_W_pos, upper_c, d_W_neg, lower_c, d_bias, pre_upper_c);
    // Same requirement for the in-place bias addition below.

    // ---------------------------------------------------------
    // [구체적 수치로 변환] 원본: pre_lower = affine_min(pre.lower_A, ...)
    // ---------------------------------------------------------
    affine_minmax_pair_gpu<<<vector_blocks, 256>>>(
        pre_lower_A, pre_lower_c, pre_upper_A, pre_upper_c,
        d_xl, d_xu, pre_lower, pre_upper);
    
    // ---------------------------------------------------------
    // [Relaxation (샌드위치 이완)] 원본: relu_relax(pre_lower, ...)
    // ---------------------------------------------------------
    if (net.act[l] == ActivationType::Relu) {
      relu_relax_full_gpu<<<vector_blocks, 256>>>(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    } else if (net.act[l] == ActivationType::Linear) {
      linear_relax_full_gpu<<<vector_blocks, 256>>>(alpha_l, beta_l, alpha_u, beta_u, weight_rows);
    } else {
      // Sigmoid는 수학이 너무 복잡해서 어쩔 수 없이 CPU로 내려서 풀고 다시 올림 (예외 상황)
      Vector h_lower, h_upper, h_alpha_l, h_beta_l, h_alpha_u, h_beta_u;
      cudaMemcpy(&h_lower, pre_lower, sizeof(Vector), cudaMemcpyDeviceToHost);
      cudaMemcpy(&h_upper, pre_upper, sizeof(Vector), cudaMemcpyDeviceToHost);
      h_lower.n = h_upper.n = weight_rows;
      sigmoid_relax(h_lower, h_upper, h_alpha_l, h_beta_l, h_alpha_u, h_beta_u);
      cudaMemcpy(alpha_l, &h_alpha_l, sizeof(Vector), cudaMemcpyHostToDevice);
      cudaMemcpy(beta_l, &h_beta_l, sizeof(Vector), cudaMemcpyHostToDevice);
      cudaMemcpy(alpha_u, &h_alpha_u, sizeof(Vector), cudaMemcpyHostToDevice);
      cudaMemcpy(beta_u, &h_beta_u, sizeof(Vector), cudaMemcpyHostToDevice);
    }

    // Preserve each layer's relaxation coefficients on the GPU so the
    // backward pass can consume them without a host round trip.
    // GPU 메모리에 계산결과 저장
    cache_relaxation_layer_gpu<<<vector_blocks, 256>>>(
        alpha_l, beta_l, alpha_u, beta_u,
        g_pool.d_fwd_alpha_lower[l], g_pool.d_fwd_beta_lower[l],
        g_pool.d_fwd_alpha_upper[l], g_pool.d_fwd_beta_upper[l]);

    // ---------------------------------------------------------
    // [Post-activation 갱신] 원본: post.lower_A = pre.lower_A * alpha_l
    // 여기서 생성된 post.lower_A를 다음 루프를 위해 current.lower_A(lower_A) 자리에 덮어쓰기
    // ---------------------------------------------------------
    rowwise_scale_pair_gpu<<<matrix_blocks, 256>>>(
        pre_lower_A, alpha_l, pre_upper_A, alpha_u, lower_A, upper_A);
    
    // 원본: post.lower_c = (pre.lower_c * alpha_l) + beta_l
    elemwise_affine_pair_gpu<<<vector_blocks, 256>>>(
        alpha_l, pre_lower_c, beta_l,
        alpha_u, pre_upper_c, beta_u,
        lower_c, upper_c);
    // 여기까지가 다음 루프를 위한 업데이트 
    //==========================================================================

    // forward 중간 결과 저장 안함 (D to H 안함) , 대신 GPU 메모리에 남아있음(backward 계산에 활용)
    if (materialize_host_results) {
      out.layer_bounds[l].dim = weight_rows;
      cudaMemcpy(&out.layer_bounds[l].alpha_lower, alpha_l, sizeof(Vector), cudaMemcpyDeviceToHost);
      cudaMemcpy(&out.layer_bounds[l].beta_lower, beta_l, sizeof(Vector), cudaMemcpyDeviceToHost);
      cudaMemcpy(&out.layer_bounds[l].alpha_upper, alpha_u, sizeof(Vector), cudaMemcpyDeviceToHost);
      cudaMemcpy(&out.layer_bounds[l].beta_upper, beta_u, sizeof(Vector), cudaMemcpyDeviceToHost);
      out.layer_bounds[l].alpha_lower.n = out.layer_bounds[l].beta_lower.n = weight_rows;
      out.layer_bounds[l].alpha_upper.n = out.layer_bounds[l].beta_upper.n = weight_rows;
    }
  }
  // ---------------------------------------------------------
  // 3. [최종 도출] 다 끝난 lower_A, lower_c 등을 최종 결과에 담아서 리턴
  // ---------------------------------------------------------

  // backward 과정에서 최종 결과 필요없으므로 D to H memcpy 안함
  if (materialize_host_results) {
    cudaMemcpy(&out.final_affine.lower_A, lower_A, sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.final_affine.upper_A, upper_A, sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.final_affine.lower_c, lower_c, sizeof(Vector), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.final_affine.upper_c, upper_c, sizeof(Vector), cudaMemcpyDeviceToHost);
    const int out_dim = network_output_dim(net);
    out.final_affine.lower_A.rows = out.final_affine.upper_A.rows = out_dim;
    out.final_affine.lower_A.cols = out.final_affine.upper_A.cols = in_dim;
    out.final_affine.lower_c.n = out.final_affine.upper_c.n = out_dim;
  
  // 최종 점수(수치) 도출
    out.final_lower = affine_min(out.final_affine.lower_A, out.final_affine.lower_c, x0, eps);
    out.final_upper = affine_max(out.final_affine.upper_A, out.final_affine.upper_c, x0, eps);
  }
  return out;
}

ForwardBoundResult lirpa_forward_bound(const FullyConnectedNetwork &net,
                                       const Vector &x0, double eps) {
  return lirpa_forward_bound_impl(net, x0, eps, true);
}

__global__ void colwise_scale_gpu(const Matrix *A, const Vector *s,
                                  Matrix *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = A->rows * A->cols;
  if (tid < total) {
    const int r = tid / A->cols;
    const int c = tid % A->cols;
    out->a[r][c] = A->a[r][c] * s->v[c];
  }
}

__global__ void split_pos_neg_gpu(const Matrix *in, Matrix *positive,
                                  Matrix *negative) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = in->rows * in->cols;
  if (tid < total) {
    const int r = tid / in->cols;
    const int c = tid % in->cols;
    const double v = in->a[r][c];
    positive->a[r][c] = v > 0.0 ? v : 0.0;
    negative->a[r][c] = v < 0.0 ? v : 0.0;
  }
}

__global__ void split_pos_neg_pair_gpu(
    const Matrix *lower_in, const Matrix *upper_in,
    Matrix *lower_pos, Matrix *lower_neg,
    Matrix *upper_pos, Matrix *upper_neg) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = lower_in->rows * lower_in->cols;
  if (tid == 0) {
    lower_pos->rows = lower_in->rows;
    lower_pos->cols = lower_in->cols;
    lower_neg->rows = lower_in->rows;
    lower_neg->cols = lower_in->cols;
    upper_pos->rows = upper_in->rows;
    upper_pos->cols = upper_in->cols;
    upper_neg->rows = upper_in->rows;
    upper_neg->cols = upper_in->cols;
  }
  if (tid < total) {
    const int r = tid / lower_in->cols;
    const int c = tid % lower_in->cols;
    const double lower_value = lower_in->a[r][c];
    const double upper_value = upper_in->a[r][c];
    lower_pos->a[r][c] = lower_value > 0.0 ? lower_value : 0.0;
    lower_neg->a[r][c] = lower_value < 0.0 ? lower_value : 0.0;
    upper_pos->a[r][c] = upper_value > 0.0 ? upper_value : 0.0;
    upper_neg->a[r][c] = upper_value < 0.0 ? upper_value : 0.0;
  }
}

__global__ void build_coeff_fused_gpu(const Matrix *in, const Vector *positive_scale,
                                      const Vector *negative_scale, Matrix *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = in->rows * in->cols;
  if (tid < total) {
    const int r = tid / in->cols;
    const int c = tid % in->cols;
    const double v = in->a[r][c];
    out->a[r][c] = v > 0.0 ? v * positive_scale->v[c]
                           : v * negative_scale->v[c];
  }
}

__global__ void build_coeff_pair_fused_gpu(
    const Matrix *lower_in, const Matrix *upper_in,
    const Vector *alpha_l, const Vector *alpha_u,
    Matrix *lower_out, Matrix *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = lower_in->rows * lower_in->cols;
  if (tid == 0) {
    lower_out->rows = lower_in->rows;
    lower_out->cols = lower_in->cols;
    upper_out->rows = upper_in->rows;
    upper_out->cols = upper_in->cols;
  }
  if (tid < total) {
    const int r = tid / lower_in->cols;
    const int c = tid % lower_in->cols;
    const double lower_value = lower_in->a[r][c];
    const double upper_value = upper_in->a[r][c];
    lower_out->a[r][c] = lower_value > 0.0
        ? lower_value * alpha_l->v[c]
        : lower_value * alpha_u->v[c];
    upper_out->a[r][c] = upper_value > 0.0
        ? upper_value * alpha_u->v[c]
        : upper_value * alpha_l->v[c];
  }
}

__global__ void affine_term_fused_gpu(const Vector *alpha, const Vector *beta,
                                      const Vector *bias, Vector *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < alpha->n) {
    out->v[tid] = alpha->v[tid] * bias->v[tid] + beta->v[tid];
  }
}

__global__ void affine_term_pair_fused_gpu(
    const Vector *alpha_l, const Vector *beta_l,
    const Vector *alpha_u, const Vector *beta_u,
    const Vector *bias, Vector *lower_out, Vector *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    lower_out->n = alpha_l->n;
    upper_out->n = alpha_l->n;
  }
  if (tid < alpha_l->n) {
    lower_out->v[tid] = alpha_l->v[tid] * bias->v[tid] + beta_l->v[tid];
    upper_out->v[tid] = alpha_u->v[tid] * bias->v[tid] + beta_u->v[tid];
  }
}

__global__ void backward_bias_fused_gpu(const Matrix *positive,
                                        const Matrix *negative,
                                        const Vector *positive_term,
                                        const Vector *negative_term,
                                        const Vector *old_p, Vector *out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < positive->rows) {
    double sum = old_p->v[tid];
    for (int j = 0; j < positive->cols; ++j) {
      sum += positive->a[tid][j] * positive_term->v[j]
           + negative->a[tid][j] * negative_term->v[j];
    }
    out->v[tid] = sum;
  }
}

__global__ void backward_bias_pair_fused_gpu(
    const Matrix *lower_pos, const Matrix *lower_neg,
    const Matrix *upper_pos, const Matrix *upper_neg,
    const Vector *lower_positive_term, const Vector *lower_negative_term,
    const Vector *upper_positive_term, const Vector *upper_negative_term,
    const Vector *lower_old_p, const Vector *upper_old_p,
    Vector *lower_out, Vector *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    lower_out->n = lower_pos->rows;
    upper_out->n = upper_pos->rows;
  }
  if (tid < lower_pos->rows) {
    double lower_sum = lower_old_p->v[tid];
    double upper_sum = upper_old_p->v[tid];
    for (int j = 0; j < lower_pos->cols; ++j) {
      lower_sum += lower_pos->a[tid][j] * lower_positive_term->v[j]
                 + lower_neg->a[tid][j] * lower_negative_term->v[j];
      upper_sum += upper_pos->a[tid][j] * upper_positive_term->v[j]
                 + upper_neg->a[tid][j] * upper_negative_term->v[j];
    }
    lower_out->v[tid] = lower_sum;
    upper_out->v[tid] = upper_sum;
  }
}

// Compute lower and upper backward matrix products with one launch.  The
// two products share the same immutable weight matrix and output coordinates.
__global__ void backward_matmul_pair_gpu(
    const Matrix *lower_in, const Matrix *upper_in, const Matrix *weight,
    Matrix *lower_out, Matrix *upper_out) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = lower_in->rows * weight->cols;
  if (tid == 0) {
    lower_out->rows = lower_in->rows;
    lower_out->cols = weight->cols;
    upper_out->rows = upper_in->rows;
    upper_out->cols = weight->cols;
  }
  if (tid < total) {
    const int r = tid / weight->cols;
    const int c = tid % weight->cols;
    double lower_sum = 0.0;
    double upper_sum = 0.0;
    for (int k = 0; k < lower_in->cols; ++k) {
      const double lower_value = lower_in->a[r][k];
      const double upper_value = upper_in->a[r][k];
      const double weight_value = weight->a[k][c];
      if (lower_value != 0.0) lower_sum += lower_value * weight_value;
      if (upper_value != 0.0) upper_sum += upper_value * weight_value;
    }
    lower_out->a[r][c] = lower_sum;
    upper_out->a[r][c] = upper_sum;
  }
}

// GPU-resident backward pass. Relaxation coefficients stay on the GPU between
// forward and backward; host copies are limited to inputs and final results.
// 단지 전체 레이어를 루프로 돌면서 커널 퓨전(Kernel Fusion)과 포인터 스왑을 통해 
// 메모리 할당/복사를 완전히 없앤 극자적 최적화(Zero-Memcpy) 버전
void backward_bound_gpu(const FullyConnectedNetwork &net,
                        const ForwardBoundResult &fwd,
                        const Matrix &initial_lower_M,
                        const Vector &initial_lower_p,
                        const Matrix &initial_upper_M,
                        const Vector &initial_upper_p,
                        const Vector &x0, double eps,
                        Vector &final_lower, Vector &final_upper,
                        Matrix &final_lower_M, Vector &final_lower_p,
                        Matrix &final_upper_M, Vector &final_upper_p,
                        bool output_matrices) {
  Matrix *lower_M = g_pool.d_bwd_mat[0];
  Matrix *upper_M = g_pool.d_bwd_mat[1];
  Matrix *lower_pos = g_pool.d_bwd_mat[2];
  Matrix *lower_neg = g_pool.d_bwd_mat[3];
  Matrix *upper_pos = g_pool.d_bwd_mat[4];
  Matrix *upper_neg = g_pool.d_bwd_mat[5];
  Matrix *lower_coeff = g_pool.d_bwd_mat[6];
  Matrix *upper_coeff = g_pool.d_bwd_mat[7];
  Matrix *new_lower_M = g_pool.d_bwd_mat[8];
  Matrix *new_upper_M = g_pool.d_bwd_mat[9];
  Vector *lower_p = g_pool.d_bwd_vec[0];
  Vector *upper_p = g_pool.d_bwd_vec[1];
  Vector *term_lp = g_pool.d_bwd_vec[6];
  Vector *term_ln = g_pool.d_bwd_vec[7];
  Vector *new_lower_p = g_pool.d_bwd_vec[10];
  Vector *new_upper_p = g_pool.d_bwd_vec[11];

  // GPU에서 사용하기 위해 초기값을 GPU로 복사
  cudaMemcpy(lower_M, &initial_lower_M, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(upper_M, &initial_upper_M, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(lower_p, &initial_lower_p, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(upper_p, &initial_upper_p, sizeof(Vector), cudaMemcpyHostToDevice);

  // Backward는 출력층(마지막 레이어)부터 입력층(0번 레이어)까지 역순으로 
  // 원래 lirpa_backward_bound 에서 루프 하던 걸 여기서 역전파 하도록 바꿈
  // 왜 이렇게 하냐면..
  for (int l = net.num_layers - 1; l >= 0; --l) {
    // The number of specification rows stays constant while propagating
    // backward; only the matrix column dimension changes per layer.
    const int m_rows = initial_lower_M.rows;
    const int m_cols = net.layer_out_dim[l];
    const int w_rows = net.layer_out_dim[l];
    const int w_cols = net.layer_in_dim[l];
    const int matrix_elems = m_rows * m_cols;
    const int matrix_blocks = (matrix_elems + 255) / 256;
    const int vector_blocks = (m_rows + 255) / 256;

    // forward에서 계산된 alpha_l, beta_l, alpha_u, beta_u (GPU 메모리에 보관하고 있던걸 가져옴)
    // 즉 backward_one_layer에서 인자로 받았던 부분을, 이미 GPU에 올려둔 부분을 사용해 해결함
    const Vector *alpha_l = g_pool.d_fwd_alpha_lower[l];
    const Vector *beta_l = g_pool.d_fwd_beta_lower[l];
    const Vector *alpha_u = g_pool.d_fwd_alpha_upper[l];
    const Vector *beta_u = g_pool.d_fwd_beta_upper[l];

    // 1. 하한(lower_M)/상한(upper_M) 행렬을 양수/음수 행렬로 분리 
    // (CPU의 positive_part, negative_part 함수 역할을 한 번에 융합 처리)
    split_pos_neg_pair_gpu<<<matrix_blocks, 256>>>(
        lower_M, upper_M, lower_pos, lower_neg, upper_pos, upper_neg);

    // 2. 분리된 행렬과 Forward에서 구한 alpha, beta를 곱해 CROWN 계수 생성
    // (CPU의 lower_s_coeff, upper_s_coeff 연산과 100% 동일)
    build_coeff_pair_fused_gpu<<<matrix_blocks, 256>>>(
        lower_M, upper_M, alpha_l, alpha_u, lower_coeff, upper_coeff);

    // 3. 계수 행렬과 가중치(Weight)를 행렬곱 연산하여 이전 레이어로 넘길 새로운 M 생성
    // (CPU의 matmul(lower_s_coeff, W)와 동일. 현재 연산 시간의 90%를 차지하는 병목 구간)
    backward_matmul_pair_gpu<<<(m_rows * w_cols + 255) / 256, 256>>>(
        lower_coeff, upper_coeff, g_pool.d_weight[l], new_lower_M,
        new_upper_M);

    // 4. 편향(bias)과 이완(relaxation) 오차(beta)를 반영하기 위한 상수항(term) 계산
    // (CPU의 vec_add(elemwise_mul(alpha, b), beta) 와 동일)
    const Vector *bias = g_pool.d_bias[l];
    affine_term_pair_fused_gpu<<<(w_rows + 255) / 256, 256>>>(
        alpha_l, beta_l, alpha_u, beta_u, bias, term_lp, term_ln);

    // 5. 앞에서 구한 행렬과 상수항을 내적(matvec)하고 누적 더하기(vec_add)를 한 번에 융합
    // (CPU의 matvec 후 vec_add 하는 복잡한 로직을 단일 커널로 압축)
    backward_bias_pair_fused_gpu<<<vector_blocks, 256>>>(
        lower_pos, lower_neg, upper_pos, upper_neg,
        term_lp, term_ln, term_ln, term_lp,
        lower_p, upper_p, new_lower_p, new_upper_p);

    // The newly computed buffers become the current state for the next
    // layer.  Swapping pointers avoids copying full matrices/vectors on GPU.
    // 6. 새로 계산된 M과 p를 다음 레이어 연산의 입력으로 사용하기 위해 포인터만 쓱 교체
    // (CPU처럼 cudaMemcpy나 새로 할당할 필요가 없는 Zero-Memcpy 아키텍처의 핵심)
    // 스왑해줘야 new_lower_M을 다음 역전파의 lower_M으로 사용할 수 있음
    std::swap(lower_M, new_lower_M);
    std::swap(upper_M, new_upper_M);
    std::swap(lower_p, new_lower_p);
    std::swap(upper_p, new_upper_p);
  }

  // =================================================================
  // [Kernel Fusion] 거대한 계수 행렬을 CPU로 가져오지 않고 GPU에서 affine_min/max 바로 계산
  // =================================================================
  
  // 1. x0, eps를 사용해 xl, xu를 CPU에서 빠르게 만듦
  Vector xl = make_zero_vector(x0.n);
  Vector xu = make_zero_vector(x0.n);
  // 여기서 vec_sub, vec_add 를 안쓰고 cpu 로 했음..
  for (int i = 0; i < x0.n; ++i) {
    xl.v[i] = x0.v[i] - eps;
    xu.v[i] = x0.v[i] + eps;
  }
  
  // 2. GPU 메모리 풀의 남는 공간을 활용해 xl, xu 업로드 (크기가 작아서 순식간)
  Vector *d_xl = g_pool.d_bwd_vec[2];
  Vector *d_xu = g_pool.d_bwd_vec[3];
  Vector *d_final_lower = g_pool.d_bwd_vec[4];
  Vector *d_final_upper = g_pool.d_bwd_vec[5];
  
  cudaMemcpy(d_xl, &xl, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(d_xu, &xu, sizeof(Vector), cudaMemcpyHostToDevice);
  
  int out_dim = initial_lower_M.rows;
  int blocksPerGrid = (out_dim + 255) / 256;
  
  // 3. GPU 안에서 최종 정답 계산
  // 커널 퓨전 가능할듯
  affine_min_gpu<<<blocksPerGrid, 256>>>(lower_M, lower_p, d_xl, d_xu, d_final_lower);
  affine_max_gpu<<<blocksPerGrid, 256>>>(upper_M, upper_p, d_xl, d_xu, d_final_upper);
  
  // 4. 길이 16짜리 최종 결과 벡터만 CPU로 D2H 복사 (통신 비용 극소화)
  final_lower = make_zero_vector(out_dim);
  final_upper = make_zero_vector(out_dim);
  cudaMemcpy(&final_lower, d_final_lower, sizeof(Vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(&final_upper, d_final_upper, sizeof(Vector), cudaMemcpyDeviceToHost);
  final_lower.n = out_dim;
  final_upper.n = out_dim;

  // 만약 사용자가 행렬 원본을 원할 경우에만(materialize_host_results=true) 복사해줌
  if (output_matrices) {
    cudaMemcpy(&final_lower_M, lower_M, sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_upper_M, upper_M, sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_lower_p, lower_p, sizeof(Vector), cudaMemcpyDeviceToHost);
    cudaMemcpy(&final_upper_p, upper_p, sizeof(Vector), cudaMemcpyDeviceToHost);
  }
}

// 이거 대신 backward_bound_gpu를 사용함 (로직은 동일, 다만 GPU, CPU구현에서 차이)
void backward_one_layer(Matrix &lower_M, Vector &lower_p, Matrix &upper_M,
                        Vector &upper_p, const Matrix &W, const Vector &b,
                        const Vector &alpha_l, const Vector &beta_l,
                        const Vector &alpha_u, const Vector &beta_u) {
  require(lower_M.cols == W.rows && upper_M.cols == W.rows,
          "backward_one_layer shape mismatch.");
  require(alpha_l.n == W.rows && beta_l.n == W.rows && alpha_u.n == W.rows &&
              beta_u.n == W.rows,
          "backward_one_layer relaxation shape mismatch.");


  // 이 부분이 1번 과정 (split_pos_neg ...)
  const Matrix lower_M_pos = positive_part(lower_M);
  const Matrix lower_M_neg = negative_part(lower_M);
  const Matrix upper_M_pos = positive_part(upper_M);
  const Matrix upper_M_neg = negative_part(upper_M);

  Matrix lower_s_coeff = make_zero_matrix(lower_M.rows, lower_M.cols);
  Matrix upper_s_coeff = make_zero_matrix(upper_M.rows, upper_M.cols);

  // lower, upper 의 조정된 계수? (아마 2번 과정..)
  for (int i = 0; i < lower_M.rows; ++i) {
    for (int j = 0; j < lower_M.cols; ++j) {
      lower_s_coeff.a[i][j] = lower_M_pos.a[i][j] * alpha_l.v[j] +
                              lower_M_neg.a[i][j] * alpha_u.v[j];
      upper_s_coeff.a[i][j] = upper_M_pos.a[i][j] * alpha_u.v[j] +
                              upper_M_neg.a[i][j] * alpha_l.v[j];
    }
  }

  // 랑 가중치를 곱하면 (3번 과정)
  const Matrix new_lower_M = matmul(lower_s_coeff, W);
  const Matrix new_upper_M = matmul(upper_s_coeff, W);

  // 4번 과정
  const Vector term_lp = vec_add(elemwise_mul(alpha_l, b), beta_l);
  const Vector term_ln = vec_add(elemwise_mul(alpha_u, b), beta_u);
  // 5번 과정
  const Vector new_lower_p = vec_add(
      vec_add(matvec(lower_M_pos, term_lp), matvec(lower_M_neg, term_ln)),
      lower_p);

  // 4번 과정
  const Vector term_up = vec_add(elemwise_mul(alpha_u, b), beta_u);
  const Vector term_un = vec_add(elemwise_mul(alpha_l, b), beta_l);
  // 5번 과정
  const Vector new_upper_p = vec_add(
      vec_add(matvec(upper_M_pos, term_up), matvec(upper_M_neg, term_un)),
      upper_p);
  // 이부분이 각각 커널 퓨전으로 통합됨

  lower_M = new_lower_M;
  lower_p = new_lower_p;
  upper_M = new_upper_M;
  upper_p = new_upper_p;
}

// materialize_forward_results : forward의 결과를 
BackwardBoundResult lirpa_backward_bound(const FullyConnectedNetwork &net, const Vector &x0,
                     double eps, bool materialize_forward_results = true,
                     const Matrix *output_lower_M = nullptr,
                     const Vector *output_lower_p = nullptr,
                     const Matrix *output_upper_M = nullptr,
                     const Vector *output_upper_p = nullptr
                     ) {

  //std::cout << "clear lirpa_forward!\n" << std::endl;
  const ForwardBoundResult fwd =
      lirpa_forward_bound_impl(net, x0, eps, materialize_forward_results);

  const int output_dim = network_output_dim(net);
  Matrix lower_M; // lower bound 선형 방정식 계수
  Matrix upper_M; // upper bound 선형 방정식 계수
  Vector lower_p; // lower bound 선형 방정식 y절편
  Vector upper_p; // upper bound 선형 방정식 y절편

  if (output_lower_M) { lower_M = *output_lower_M; }
  else { lower_M = make_eye(output_dim); }
  if (output_upper_M) { upper_M = *output_upper_M; }
  else { upper_M = make_eye(output_dim); }
  if (output_lower_p) { lower_p = *output_lower_p; }
  else { lower_p = make_zero_vector(lower_M.rows); }
  if (output_upper_p) { upper_p = *output_upper_p; }
  else { upper_p = make_zero_vector(upper_M.rows); }

  require(lower_M.cols == output_dim && upper_M.cols == output_dim,
          "Output spec matrix column mismatch.");
  require(lower_M.rows == upper_M.rows,
          "Output spec lower/upper row mismatch.");
  require(lower_p.n == lower_M.rows && upper_p.n == upper_M.rows,
          "Output spec vector row mismatch.");

  BackwardBoundResult out;
  
  // 과거에는 바로 이 위치에서 for (int l = net.num_layers - 1; l >= 0; --l) 
  // 루프를 돌면서 backward_one_layer(...) 를 층(layer) 개수만큼 반복 호출
  // 내부적으로 커널 퓨전(Kernel Fusion)과 Zero-Memcpy 루프가 적용된
  // 단일 함수 backward_bound_gpu()를 딱 한 번만 호출하여 병목 제거

  // 초기 lower_M,p / upper_M,p를 준다.
  // 나중에 lower_M,p / upper_M,p ==> out.final에 값 덮어쓰기
  // 커널 퓨전 덕분에 out.final_lower, upper 까지 여기서 한번에 계산됨
  backward_bound_gpu(net, fwd, lower_M, lower_p, upper_M, upper_p,
                     x0, eps, out.final_lower, out.final_upper,
                     out.final_affine.lower_A, out.final_affine.lower_c,
                     out.final_affine.upper_A, out.final_affine.upper_c,
                     materialize_forward_results);

  out.num_layer_bounds = fwd.num_layer_bounds;

  // forward에서 얻은 중간 계산 결과 스킵
  if (materialize_forward_results) {
    for (int i = 0; i < fwd.num_layer_bounds; ++i) {
      out.layer_bounds[i] = fwd.layer_bounds[i];
    }
  }
  return out;
}

// 얘는 재귀버전이라 CUDA로 재작성하기 쉽지 않을 것임.. ==> 이터레이션 버전 추가
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

  static LayerBound build_one_layer_relaxation(const FullyConnectedNetwork &net, int layer,
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

// ============================================================
// XOR 데모 및 테스트 (my_lirpa.cu와 동일)
// ============================================================

FullyConnectedNetwork make_xor_network() {
  int layer_in[MAX_LAYERS]{};
  int layer_out[MAX_LAYERS]{};
  double W_data[MAX_LAYERS][MAX_DIM][MAX_DIM]{};
  double b_data[MAX_LAYERS][MAX_DIM]{};
  std::string acts[MAX_LAYERS];

  const int L = 2;
  layer_in[0] = 2;  layer_out[0] = 2;  acts[0] = "relu";
  layer_in[1] = 2;  layer_out[1] = 1;  acts[1] = "sigmoid";

  W_data[0][0][0] = 2.1247;  W_data[0][0][1] = 2.1267;
  W_data[0][1][0] = -2.1237; W_data[0][1][1] = -2.1235;
  b_data[0][0] = -2.1259;    b_data[0][1] = 2.1234;

  W_data[1][0][0] = -3.6788; W_data[1][0][1] = -3.6766;
  b_data[1][0] = 3.5451;

  return make_network(L, layer_in, layer_out, W_data, b_data, acts);
}

// 아래 3개의 함수는 테스트용 (임시 데이터셋 + temp 신경망)
void self_test_relaxations() {
  std::mt19937_64 rng(0);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  for (int t = 0; t < 200; ++t) {
    double a = dist(rng);
    double b = dist(rng);
    if (a > b) std::swap(a, b);
    if (std::abs(a - b) < 1e-8) b = a + 1e-6;

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

int xor_expected_label(const Vector &x) {
  require(x.n >= 2, "xor_expected_label requires at least 2-dimensional input.");
  const int a = static_cast<int>(std::llround(x.v[0]));
  const int b = static_cast<int>(std::llround(x.v[1]));
  return a ^ b;
}

void run_xor_demo(double eps) {
  const FullyConnectedNetwork network = make_xor_network();
  const LiRPABackwardOnly backward_only_verifier;

  Vector points[4];
  for (int i = 0; i < 4; ++i) { points[i] = make_zero_vector(2); }
  points[0].v[0] = 0.0; points[0].v[1] = 0.0;
  points[1].v[0] = 0.0; points[1].v[1] = 1.0;
  points[2].v[0] = 1.0; points[2].v[1] = 0.0;
  points[3].v[0] = 1.0; points[3].v[1] = 1.0;

  std::cout << "XOR network point predictions and LiRPA-certified output bounds\n";
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

    bool fwd_certified, bwd_certified, bwd_only_certified;
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
    all_certified_backward_only = all_certified_backward_only && bwd_only_certified;

    std::cout << "x0=[" << x0.v[0] << ", " << x0.v[1]
              << "], expected=" << expected << ", network_output=" << y.v[0] << "\n";
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
  if (all_certified_forward)
    std::cout << "Forward mode certifies all four XOR corner classifications for this epsilon.\n";
  else
    std::cout << "Forward mode does not certify at least one XOR corner classification for this epsilon.\n";

  if (all_certified_backward)
    std::cout << "Backward mode certifies all four XOR corner classifications for this epsilon.\n";
  else
    std::cout << "Backward mode does not certify at least one XOR corner classification for this epsilon.\n";

  if (all_certified_backward_only)
    std::cout << "Backward-only mode certifies all four XOR corner classifications for this epsilon.\n";
  else
    std::cout << "Backward-only mode does not certify at least one XOR corner classification for this epsilon.\n";
}

} // namespace
