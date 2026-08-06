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
  int rows = 0;
  int cols = 0;
  double a[MAX_DIM][MAX_DIM]{};
};

struct Vector {
  int n = 0;
  double v[MAX_DIM]{};
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
  LayerBound layer_bounds[MAX_LAYERS]{};
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
// 메모리 풀 (Memory Pool) - 이 파일의 유일한 변경점
// ============================================================
// 함수들이 사용하는 최대 동시 GPU 버퍼 수:
//   Matrix: 3개 (mat_add: A,B,C / matmul: A,B,C / rowwise_scale: A,out + vec1)
//   Vector: 6개 (relu_relax: lower,upper,alpha_l,beta_l,alpha_u,beta_u)
// affine_min/max는 1 Matrix + 5 Vector이므로 위 범위 안에 들어감

constexpr int NUM_POOL_MAT = 3;
constexpr int NUM_POOL_VEC = 6;

struct GpuPool {
  Matrix* d_mat[NUM_POOL_MAT];
  Vector* d_vec[NUM_POOL_VEC];
  bool initialized = false;

  // 1번만 실행되도록
  void init() {
    if (initialized) return;
    for (int i = 0; i < NUM_POOL_MAT; i++)
      cudaMalloc(&d_mat[i], sizeof(Matrix));
    for (int i = 0; i < NUM_POOL_VEC; i++)
      cudaMalloc(&d_vec[i], sizeof(Vector));
    initialized = true;
  }

  void destroy() {
    if (!initialized) return;
    for (int i = 0; i < NUM_POOL_MAT; i++)
      cudaFree(d_mat[i]);
    for (int i = 0; i < NUM_POOL_VEC; i++)
      cudaFree(d_vec[i]);
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

// --- CPU 원본 함수들 (성능 비교 및 백업용) ---
Matrix make_zero_matrix_cpu(int rows, int cols) {
  require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM,
          "Matrix shape out of bounds.");
  Matrix out;
  out.rows = rows;
  out.cols = cols;
  return out;
}

Vector make_zero_vector_cpu(int n) {
  require(n >= 0 && n <= MAX_DIM, "Vector length out of bounds.");
  Vector out;
  out.n = n;
  return out;
}

Matrix make_eye_cpu(int n) {
  Matrix out = make_zero_matrix_cpu(n, n);
  for (int i = 0; i < n; ++i) {
    out.a[i][i] = 1.0;
  }
  return out;
}

// ============================================================
// make 함수들 - 풀 사용으로 변경
// ============================================================

Matrix make_zero_matrix(int rows, int cols) {
  require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM,
          "Matrix shape out of bounds.");

  // 메모리 할당 (할당되어 있으면 바로 리턴함)
  ensure_pool();
  Matrix out;

  cudaMemset(g_pool.d_mat[0], 0, sizeof(Matrix));
  cudaMemcpy(&out, g_pool.d_mat[0], sizeof(Matrix), cudaMemcpyDeviceToHost);
  out.rows = rows;
  out.cols = cols;

  return out;
}

Vector make_zero_vector(int n) {
  require(n >= 0 && n <= MAX_DIM, "Vector length out of bounds.");
  ensure_pool();
  Vector out;

  cudaMemset(g_pool.d_vec[0], 0, sizeof(Vector));
  cudaMemcpy(&out, g_pool.d_vec[0], sizeof(Vector), cudaMemcpyDeviceToHost);
  out.n = n;

  return out;
}

__global__ void make_eye_gpu(Matrix *out, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    out->a[tid][tid] = 1.0;
  }
}

Matrix make_eye(int n) {
  ensure_pool();
  Matrix out = make_zero_matrix(n, n);

  cudaMemcpy(g_pool.d_mat[0], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid == 0) blocksPerGrid = 1;

  make_eye_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], n);
  cudaDeviceSynchronize();

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
  cudaMemcpy(g_pool.d_mat[2], &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  mat_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1], g_pool.d_mat[2]);
  cudaDeviceSynchronize();

  cudaMemcpy(&C, g_pool.d_mat[2], sizeof(Matrix), cudaMemcpyDeviceToHost);
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
  cudaMemcpy(g_pool.d_vec[2], &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  vec_add_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2]);
  cudaDeviceSynchronize();

  cudaMemcpy(&c, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
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
  cudaMemcpy(g_pool.d_vec[2], &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  vec_sub_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2]);
  cudaDeviceSynchronize();

  cudaMemcpy(&c, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
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
  cudaMemcpy(g_pool.d_mat[2], &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * B.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  matmul_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1], g_pool.d_mat[2]);
  cudaDeviceSynchronize();

  cudaMemcpy(&C, g_pool.d_mat[2], sizeof(Matrix), cudaMemcpyDeviceToHost);
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

Vector matvec(const Matrix &A, const Vector &x) {
  require(A.cols == x.n, "matvec shape mismatch.");
  ensure_pool();
  Vector y = make_zero_vector(A.rows);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[0], &x, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[1], &y, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  matvec_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0], g_pool.d_vec[1]);
  cudaDeviceSynchronize();

  cudaMemcpy(&y, g_pool.d_vec[1], sizeof(Vector), cudaMemcpyDeviceToHost);
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

Matrix positive_part(const Matrix &A) {
  ensure_pool();
  Matrix out = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_mat[1], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  positive_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1]);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_mat[1], sizeof(Matrix), cudaMemcpyDeviceToHost);
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

Matrix negative_part(const Matrix &A) {
  ensure_pool();
  Matrix out = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_mat[1], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  negative_part_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_mat[1]);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_mat[1], sizeof(Matrix), cudaMemcpyDeviceToHost);
  return out;
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

Matrix rowwise_scale(const Matrix &A, const Vector &s) {
  require(A.rows == s.n, "rowwise_scale shape mismatch.");
  ensure_pool();
  Matrix out = make_zero_matrix(A.rows, A.cols);

  cudaMemcpy(g_pool.d_mat[0], &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_vec[0], &s, sizeof(Vector), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_mat[1], &out, sizeof(Matrix), cudaMemcpyHostToDevice);

  int total_elements = A.rows * A.cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

  rowwise_scale_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0], g_pool.d_mat[1]);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_mat[1], sizeof(Matrix), cudaMemcpyDeviceToHost);
  return out;
}

// elemwise_mul (Vector 3개 사용)
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
  cudaMemcpy(g_pool.d_vec[2], &c, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (a.n + threadsPerBlock - 1) / threadsPerBlock;

  elemwise_mul_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], g_pool.d_vec[1], g_pool.d_vec[2]);
  cudaDeviceSynchronize();

  cudaMemcpy(&c, g_pool.d_vec[2], sizeof(Vector), cudaMemcpyDeviceToHost);
  return c;
}

// CPU 원본 (비교용)
Vector make_eps_vec_cpu(int n, double eps) {
  Vector out = make_zero_vector_cpu(n);
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

  cudaMemcpy(g_pool.d_vec[0], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  if (blocksPerGrid == 0) blocksPerGrid = 1;

  make_eps_vec_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_vec[0], n, eps);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_vec[0], sizeof(Vector), cudaMemcpyDeviceToHost);
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
  cudaMemcpy(g_pool.d_vec[3], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  affine_min_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0],
                                                     g_pool.d_vec[1], g_pool.d_vec[2],
                                                     g_pool.d_vec[3]);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_vec[3], sizeof(Vector), cudaMemcpyDeviceToHost);
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
  cudaMemcpy(g_pool.d_vec[3], &out, sizeof(Vector), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (A.rows + threadsPerBlock - 1) / threadsPerBlock;

  affine_max_gpu<<<blocksPerGrid, threadsPerBlock>>>(g_pool.d_mat[0], g_pool.d_vec[0],
                                                     g_pool.d_vec[1], g_pool.d_vec[2],
                                                     g_pool.d_vec[3]);
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_vec[3], sizeof(Vector), cudaMemcpyDeviceToHost);
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

// 2개의 벡터를 입력으로 사용 (lower, upper)
// 4개의 벡터를 출력으로 사용 (alpha_l, beta_l, alpha_u, beta_u)
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
  cudaDeviceSynchronize();

  cudaMemcpy(&out, g_pool.d_vec[0], sizeof(Vector), cudaMemcpyDeviceToHost);
  return out;
}

Vector network_forward(const FullyConnectedNetwork &net, const Vector &x) {
  require(x.n == network_input_dim(net),
          "network_forward input dimension mismatch.");
  Vector f = x;
  for (int l = 0; l < net.num_layers; ++l) {
    Vector s = vec_add(matvec(net.W[l], f), net.b[l]);
    f = apply_activation(s, net.act[l]);
  }
  return f;
}

// ============================================================
// CROWN 알고리즘 (my_lirpa.cu와 동일한 로직)
// ============================================================

ForwardBoundResult lirpa_forward_bound(const FullyConnectedNetwork &net,
                                       const Vector &x0, double eps) {
  require(x0.n == network_input_dim(net),
          "lirpa_forward_bound input dimension mismatch.");

  const int in_dim = network_input_dim(net); // 입력 차원(노드) 개수 
  
  // 1. [초기화 단계] 입력층의 수식을 항등 행렬(Identity)과 0 벡터로 시작합니다.
  // 수식으로 치면 f(x) = 1*x + 0 과 같습니다.
  AffineBound current;                       
  current.lower_A = make_eye(in_dim);        // 하한선 계수 행렬 A (단위 행렬)
  current.upper_A = make_eye(in_dim);        // 상한선 계수 행렬 A (단위 행렬)
  current.lower_c = make_zero_vector(in_dim); // 하한선 편향 벡터 c (0 벡터)
  current.upper_c = make_zero_vector(in_dim); // 상한선 편향 벡터 c (0 벡터)

  ForwardBoundResult out;
  out.num_layer_bounds = net.num_layers;

  // 2. [레이어 순회] 입력층부터 출력층까지 순차적으로(Forward) 수식을 밀어냅니다.
  for (int l = 0; l < net.num_layers; ++l) {
    // 가중치 행렬 W를 양수(W_pos)와 음수(W_neg) 파트로 분리합니다.
    // 이유: 부등식에서 음수를 곱하면 부등호 방향(상한/하한)이 뒤집히기 때문입니다.
    const Matrix W_pos = positive_part(net.W[l]);
    const Matrix W_neg = negative_part(net.W[l]);

    // [Pre-activation 계산] 활성화 함수(ReLU)를 거치기 전의 선형 수식(Wx+b)을 구합니다.
    AffineBound pre;
    // 하한선의 계수: 양수 가중치에는 하한선을 곱하고, 음수 가중치에는 상한선을 곱해 더합니다 (최악의 하한선)
    pre.lower_A =
        mat_add(matmul(W_pos, current.lower_A), matmul(W_neg, current.upper_A));
    pre.lower_c = vec_add(
        vec_add(matvec(W_pos, current.lower_c), matvec(W_neg, current.upper_c)),
        net.b[l]);

    // 상한선의 계수: 양수 가중치에는 상한선을 곱하고, 음수 가중치에는 하한선을 곱해 더합니다 (최상의 상한선)
    pre.upper_A =
        mat_add(matmul(W_pos, current.upper_A), matmul(W_neg, current.lower_A));
    pre.upper_c = vec_add(
        vec_add(matvec(W_pos, current.upper_c), matvec(W_neg, current.lower_c)),
        net.b[l]);

    // 위에서 구한 선형 수식(계수 A와 편향 c)을 바탕으로, 
    // 입력 공간(x0 ± eps) 내에서 가질 수 있는 실제 최솟값(pre_lower)과 최댓값(pre_upper)을 계산합니다.
    const Vector pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
    const Vector pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

    // [Relaxation (이완) 계산] 비선형 함수(ReLU 등)를 선형 부등식 두 개(상한/하한 직선)로 근사(Relax)합니다.
    Vector alpha_l, beta_l, alpha_u, beta_u;
    if (net.act[l] == ActivationType::Linear) { // 선형일 때는 그대로 1차 함수 유지
      alpha_l = make_zero_vector(pre_lower.n);
      alpha_u = make_zero_vector(pre_lower.n);
      beta_l = make_zero_vector(pre_lower.n);
      beta_u = make_zero_vector(pre_lower.n);
      for (int i = 0; i < pre_lower.n; ++i) {
        alpha_l.v[i] = 1.0;
        alpha_u.v[i] = 1.0;
      }
    } else if (net.act[l] == ActivationType::Relu) {
      // ReLU 곡선을 덮어씌우는 두 개의 직선(기울기 alpha, 절편 beta)을 계산합니다.
      relu_relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    } else {
      sigmoid_relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
    }

    // [Post-activation 계산] 방금 구한 이완 직선(alpha, beta)을 이전 수식(pre)에 곱해서 더합니다.
    AffineBound post;
    post.lower_A = rowwise_scale(pre.lower_A, alpha_l); // 계수행렬 A에 기울기 alpha_l을 곱함
    post.lower_c = vec_add(elemwise_mul(alpha_l, pre.lower_c), beta_l); // 편향 c에 alpha_l을 곱하고 beta_l을 더함
    post.upper_A = rowwise_scale(pre.upper_A, alpha_u); // 상한선도 동일하게 진행
    post.upper_c = vec_add(elemwise_mul(alpha_u, pre.upper_c), beta_u);

    // 이제 현재(current) 상태를 갱신하고 다음 레이어로 넘어갈 준비를 합니다.
    current = post;

    // 나중에 Backward(역방향) 계산에서 이 값들을 다시 써먹어야 하므로 캐싱해둡니다.
    out.layer_bounds[l].dim = alpha_l.n;
    out.layer_bounds[l].alpha_lower = alpha_l;
    out.layer_bounds[l].beta_lower = beta_l;
    out.layer_bounds[l].alpha_upper = alpha_u;
    out.layer_bounds[l].beta_upper = beta_u;
  }

  // 3. [최종 결과 도출] 모든 레이어를 통과한 최종 수식(최종 계수 A와 편향 c)
  out.final_affine = current;
  // 그 최종 수식에 입력 박스(x0 ± eps)를 대입하여 절대적인 최종 하한/상한 범위를 확정합니다.
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

BackwardBoundResult lirpa_backward_bound(const FullyConnectedNetwork &net, const Vector &x0,
                     double eps, const Matrix *output_lower_M = nullptr,
                     const Vector *output_lower_p = nullptr,
                     const Matrix *output_upper_M = nullptr,
                     const Vector *output_upper_p = nullptr) {

  //std::cout << "clear lirpa_forward!\n" << std::endl;
  const ForwardBoundResult fwd = lirpa_forward_bound(net, x0, eps);

  const int output_dim = network_output_dim(net);
  Matrix lower_M;
  Matrix upper_M;
  Vector lower_p;
  Vector upper_p;

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
