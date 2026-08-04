// ============================================================
// my_lirpa_op_hybrid.cu - 하이브리드 최적화 CROWN/LiRPA
//
// 전략: matmul(O(n³))만 GPU(cuBLAS), 나머지 전부 CPU
// - mat_add, positive_part, negative_part 등 O(n²) 연산은
//   cudaMemcpy(2MB) 왕복 비용이 CPU 직접 계산보다 비싸므로 CPU 처리
// - matmul만 cuBLAS로 가속 (연산량이 O(n³)이라 GPU가 확실히 유리)
//
// 컴파일: nvcc -Xcompiler "/utf-8" -Xlinker "/STACK:134217728"
//         -arch=sm_89 -lcublas crown_test_op_hybrid.cu -o crown_test_op_hybrid.exe
// ============================================================

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

// ============================================================
// 1. 상수 및 구조체 정의
// ============================================================

constexpr int MAX_LAYERS = 16;
constexpr int MAX_DIM = 512;

enum class ActivationType { Relu, Sigmoid, Linear };

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
// 2. GPU 리소스 (matmul 전용 - 최소한의 GPU 사용)
// ============================================================

struct GpuMatmulPool {
  Matrix* d_A;
  Matrix* d_B;
  Matrix* d_C;
  cublasHandle_t cublas;
  bool initialized = false;

  void init() {
    if (initialized) return;
    cudaMalloc(&d_A, sizeof(Matrix));
    cudaMalloc(&d_B, sizeof(Matrix));
    cudaMalloc(&d_C, sizeof(Matrix));
    cublasCreate(&cublas);
    initialized = true;
  }

  void destroy() {
    if (!initialized) return;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(cublas);
    initialized = false;
  }
};

static GpuMatmulPool g_pool;

void ensure_pool() {
  if (!g_pool.initialized) g_pool.init();
}

// ============================================================
// 3. 기본 유틸리티 (CPU)
// ============================================================

inline double pos(double x) { return std::max(x, 0.0); }
inline double neg(double x) { return std::min(x, 0.0); }
inline double relu(double x) { return std::max(x, 0.0); }

__host__ __device__ inline double sigmoid(double x) {
  if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
  const double ex = exp(x);
  return ex / (1.0 + ex);
}

__host__ __device__ inline double sigmoid_prime(double x) {
  const double s = sigmoid(x);
  return s * (1.0 - s);
}

void require(bool cond, const std::string &msg) {
  if (!cond) throw std::invalid_argument(msg);
}

// ============================================================
// 4. 전부 CPU 연산 (matmul 제외)
// ============================================================

Matrix make_zero_matrix(int rows, int cols) {
  require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM,
          "Matrix shape out of bounds.");
  Matrix out;
  out.rows = rows;
  out.cols = cols;
  return out;
}

Vector make_zero_vector(int n) {
  require(n >= 0 && n <= MAX_DIM, "Vector length out of bounds.");
  Vector out;
  out.n = n;
  return out;
}

Matrix make_eye(int n) {
  Matrix out = make_zero_matrix(n, n);
  for (int i = 0; i < n; ++i) out.a[i][i] = 1.0;
  return out;
}

// --- O(n²) 행렬 연산: 전부 CPU (cudaMemcpy 2MB 왕복보다 빠름) ---

Matrix mat_add(const Matrix &A, const Matrix &B) {
  require(A.rows == B.rows && A.cols == B.cols, "mat_add shape mismatch.");
  Matrix C = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < A.cols; ++j)
      C.a[i][j] = A.a[i][j] + B.a[i][j];
  return C;
}

Matrix positive_part(const Matrix &A) {
  Matrix out = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < A.cols; ++j)
      out.a[i][j] = pos(A.a[i][j]);
  return out;
}

Matrix negative_part(const Matrix &A) {
  Matrix out = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < A.cols; ++j)
      out.a[i][j] = neg(A.a[i][j]);
  return out;
}

Matrix rowwise_scale(const Matrix &A, const Vector &s) {
  require(A.rows == s.n, "rowwise_scale shape mismatch.");
  Matrix out = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < A.cols; ++j)
      out.a[i][j] = A.a[i][j] * s.v[i];
  return out;
}

// --- O(n) 벡터 연산: CPU ---

Vector vec_add(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_add shape mismatch.");
  Vector out = make_zero_vector(a.n);
  for (int i = 0; i < a.n; ++i) out.v[i] = a.v[i] + b.v[i];
  return out;
}

Vector vec_sub(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_sub shape mismatch.");
  Vector out = make_zero_vector(a.n);
  for (int i = 0; i < a.n; ++i) out.v[i] = a.v[i] - b.v[i];
  return out;
}

Vector matvec(const Matrix &A, const Vector &x) {
  require(A.cols == x.n, "matvec shape mismatch.");
  Vector y = make_zero_vector(A.rows);
  for (int i = 0; i < A.rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < A.cols; ++j) sum += A.a[i][j] * x.v[j];
    y.v[i] = sum;
  }
  return y;
}

Vector elemwise_mul(const Vector &a, const Vector &b) {
  require(a.n == b.n, "elemwise_mul shape mismatch.");
  Vector out = make_zero_vector(a.n);
  for (int i = 0; i < a.n; ++i) out.v[i] = a.v[i] * b.v[i];
  return out;
}

Vector make_eps_vec(int n, double eps) {
  Vector out = make_zero_vector(n);
  for (int i = 0; i < n; ++i) out.v[i] = eps;
  return out;
}

Vector affine_min(const Matrix &A, const Vector &c, const Vector &x0, double eps) {
  require(A.rows == c.n && A.cols == x0.n, "affine_min shape mismatch.");
  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);
  Vector out = make_zero_vector(A.rows);
  for (int i = 0; i < A.rows; ++i) {
    double sum = c.v[i];
    for (int j = 0; j < A.cols; ++j)
      sum += pos(A.a[i][j]) * xl.v[j] + neg(A.a[i][j]) * xu.v[j];
    out.v[i] = sum;
  }
  return out;
}

Vector affine_max(const Matrix &A, const Vector &c, const Vector &x0, double eps) {
  require(A.rows == c.n && A.cols == x0.n, "affine_max shape mismatch.");
  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);
  Vector out = make_zero_vector(A.rows);
  for (int i = 0; i < A.rows; ++i) {
    double sum = c.v[i];
    for (int j = 0; j < A.cols; ++j)
      sum += pos(A.a[i][j]) * xu.v[j] + neg(A.a[i][j]) * xl.v[j];
    out.v[i] = sum;
  }
  return out;
}

// ============================================================
// 5. matmul만 GPU (cuBLAS) - 유일한 GPU 연산!
// ============================================================

Matrix matmul(const Matrix &A, const Matrix &B) {
  require(A.cols == B.rows, "matmul shape mismatch.");
  ensure_pool();

  Matrix C = make_zero_matrix(A.rows, B.cols);

  // 풀 버퍼에 복사 (cudaMalloc 없음!)
  cudaMemcpy(g_pool.d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_B, &B, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(g_pool.d_C, &C, sizeof(Matrix), cudaMemcpyHostToDevice);

  // Matrix 구조체 내 double 배열의 오프셋
  size_t a_offset = offsetof(Matrix, a);
  double* d_A_data = (double*)((char*)g_pool.d_A + a_offset);
  double* d_B_data = (double*)((char*)g_pool.d_B + a_offset);
  double* d_C_data = (double*)((char*)g_pool.d_C + a_offset);

  double alpha = 1.0, beta = 0.0;

  // Row-major 트릭: C = A * B (row-major) ↔ C^T = B^T * A^T (col-major)
  cublasDgemm(g_pool.cublas,
              CUBLAS_OP_N, CUBLAS_OP_N,
              B.cols, A.rows, A.cols,
              &alpha,
              d_B_data, MAX_DIM,
              d_A_data, MAX_DIM,
              &beta,
              d_C_data, MAX_DIM);
  cudaDeviceSynchronize();

  cudaMemcpy(&C, g_pool.d_C, sizeof(Matrix), cudaMemcpyDeviceToHost);
  return C;
}

// ============================================================
// 6. Relaxation 함수 (CPU)
// ============================================================

void relu_relax(const Vector &lower, const Vector &upper, Vector &alpha_l,
                Vector &beta_l, Vector &alpha_u, Vector &beta_u) {
  require(lower.n == upper.n, "relu_relax shape mismatch.");
  alpha_l = make_zero_vector(lower.n);
  beta_l = make_zero_vector(lower.n);
  alpha_u = make_zero_vector(lower.n);
  beta_u = make_zero_vector(lower.n);

  for (int i = 0; i < lower.n; ++i) {
    const double l = lower.v[i];
    const double u = upper.v[i];
    require(l <= u, "Invalid interval in relu_relax.");
    if (l >= 0.0) {
      alpha_l.v[i] = 1.0;
      alpha_u.v[i] = 1.0;
    } else if (u <= 0.0) {
      // zero
    } else {
      const double denom = u - l;
      alpha_u.v[i] = u / denom;
      beta_u.v[i] = -u * l / denom;
      alpha_l.v[i] = (std::abs(l) < std::abs(u)) ? 1.0 : 0.0;
    }
  }
}

double bisect_root(double lo, double hi,
                   const std::function<double(double)> &fn,
                   int max_iter = 80, double tol = 1e-12) {
  double flo = fn(lo), fhi = fn(hi);
  if (std::abs(flo) < tol) return lo;
  if (std::abs(fhi) < tol) return hi;

  if (flo * fhi > 0.0) {
    constexpr int GRID = 257;
    std::array<double, GRID> xs{}, vals{};
    for (int i = 0; i < GRID; ++i) {
      xs[i] = lo + (hi - lo) * (double)i / (double)(GRID - 1);
      vals[i] = fn(xs[i]);
    }
    int best = 0;
    double best_abs = std::abs(vals[0]);
    for (int i = 1; i < GRID; ++i) {
      double cur = std::abs(vals[i]);
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
    double mid = 0.5 * (lo + hi);
    double fmid = fn(mid);
    if (std::abs(fmid) < tol || std::abs(hi - lo) < tol) return mid;
    if (flo * fmid <= 0.0) hi = mid;
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
      double slope = sigmoid_prime(l);
      double intercept = sigmoid(l) - slope * l;
      alpha_l.v[i] = slope; beta_l.v[i] = intercept;
      alpha_u.v[i] = slope; beta_u.v[i] = intercept;
      continue;
    }

    if (l >= 0.0) {
      double slope_sec = (sigmoid(u) - sigmoid(l)) / (u - l);
      alpha_l.v[i] = slope_sec;
      beta_l.v[i] = sigmoid(u) - slope_sec * u;
      double x0 = 0.5 * (l + u);
      double slope_tan = sigmoid_prime(x0);
      alpha_u.v[i] = slope_tan;
      beta_u.v[i] = sigmoid(x0) - slope_tan * x0;
    } else if (u <= 0.0) {
      double x0 = 0.5 * (l + u);
      double slope_tan = sigmoid_prime(x0);
      alpha_l.v[i] = slope_tan;
      beta_l.v[i] = sigmoid(x0) - slope_tan * x0;
      double slope_sec = (sigmoid(u) - sigmoid(l)) / (u - l);
      alpha_u.v[i] = slope_sec;
      beta_u.v[i] = sigmoid(u) - slope_sec * u;
    } else {
      double su = sigmoid(u);
      auto fn_lower = [su, u](double d) {
        return (su - sigmoid(d)) / (u - d) - sigmoid_prime(d);
      };
      double du = bisect_root(l, 0.0, fn_lower);
      double sl = sigmoid(l);
      auto fn_upper = [sl, l](double d) {
        return (sigmoid(d) - sl) / (d - l) - sigmoid_prime(d);
      };
      double dl = bisect_root(0.0, u, fn_upper);
      double slope_lower = sigmoid_prime(du);
      alpha_l.v[i] = slope_lower;
      beta_l.v[i] = sigmoid(du) - slope_lower * du;
      double slope_upper = sigmoid_prime(dl);
      alpha_u.v[i] = slope_upper;
      beta_u.v[i] = sigmoid(dl) - slope_upper * dl;
    }

    double lower_violation = 0.0, upper_violation = 0.0;
    constexpr int SAMPLES = 1001;
    for (int k = 0; k < SAMPLES; ++k) {
      double x = l + (u - l) * (double)k / (double)(SAMPLES - 1);
      double y = sigmoid(x);
      lower_violation = std::max(lower_violation, alpha_l.v[i] * x + beta_l.v[i] - y);
      upper_violation = std::max(upper_violation, y - (alpha_u.v[i] * x + beta_u.v[i]));
    }
    if (lower_violation > 1e-10) beta_l.v[i] -= lower_violation + 1e-10;
    if (upper_violation > 1e-10) beta_u.v[i] += upper_violation + 1e-10;
  }
}

// ============================================================
// 7. 신경망 관련 함수
// ============================================================

ActivationType parse_activation(const std::string &name) {
  std::string lower = name;
  for (char &ch : lower) ch = (char)std::tolower((unsigned char)ch);
  if (lower == "relu") return ActivationType::Relu;
  if (lower == "sigmoid") return ActivationType::Sigmoid;
  if (lower == "linear") return ActivationType::Linear;
  throw std::invalid_argument("Unsupported activation: " + name);
}

int network_input_dim(const FullyConnectedNetwork &net) {
  return net.layer_in_dim[0];
}

int network_output_dim(const FullyConnectedNetwork &net) {
  return net.layer_out_dim[net.num_layers - 1];
}

Vector network_forward(const FullyConnectedNetwork &net, const Vector &x) {
  require(x.n == network_input_dim(net), "network_forward input dimension mismatch.");
  Vector f = x;
  for (int l = 0; l < net.num_layers; ++l) {
    Vector s = vec_add(matvec(net.W[l], f), net.b[l]);
    for (int i = 0; i < s.n; ++i) {
      if (net.act[l] == ActivationType::Relu) s.v[i] = relu(s.v[i]);
      else if (net.act[l] == ActivationType::Sigmoid) s.v[i] = sigmoid(s.v[i]);
    }
    f = s;
  }
  return f;
}

// ============================================================
// 8. CROWN 알고리즘 (matmul만 cuBLAS, 나머지 CPU)
// ============================================================

ForwardBoundResult lirpa_forward_bound(const FullyConnectedNetwork &net,
                                       const Vector &x0, double eps) {
  require(x0.n == network_input_dim(net), "lirpa_forward_bound input dimension mismatch.");

  const int in_dim = network_input_dim(net);
  AffineBound current;
  current.lower_A = make_eye(in_dim);
  current.upper_A = make_eye(in_dim);
  current.lower_c = make_zero_vector(in_dim);
  current.upper_c = make_zero_vector(in_dim);

  ForwardBoundResult out;
  out.num_layer_bounds = net.num_layers;

  for (int l = 0; l < net.num_layers; ++l) {
    const Matrix W_pos = positive_part(net.W[l]);  // CPU
    const Matrix W_neg = negative_part(net.W[l]);  // CPU

    AffineBound pre;
    // matmul만 GPU(cuBLAS)!
    pre.lower_A = mat_add(matmul(W_pos, current.lower_A), matmul(W_neg, current.upper_A));
    pre.lower_c = vec_add(
        vec_add(matvec(W_pos, current.lower_c), matvec(W_neg, current.upper_c)),
        net.b[l]);
    pre.upper_A = mat_add(matmul(W_pos, current.upper_A), matmul(W_neg, current.lower_A));
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
    post.lower_A = rowwise_scale(pre.lower_A, alpha_l);  // CPU
    post.lower_c = vec_add(elemwise_mul(alpha_l, pre.lower_c), beta_l);
    post.upper_A = rowwise_scale(pre.upper_A, alpha_u);  // CPU
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
  require(alpha_l.n == W.rows && beta_l.n == W.rows &&
          alpha_u.n == W.rows && beta_u.n == W.rows,
          "backward_one_layer relaxation shape mismatch.");

  const Matrix lower_M_pos = positive_part(lower_M);  // CPU
  const Matrix lower_M_neg = negative_part(lower_M);  // CPU
  const Matrix upper_M_pos = positive_part(upper_M);  // CPU
  const Matrix upper_M_neg = negative_part(upper_M);  // CPU

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

  // matmul만 GPU(cuBLAS)!
  const Matrix new_lower_M = matmul(lower_s_coeff, W);
  const Matrix new_upper_M = matmul(upper_s_coeff, W);

  const Vector term_lp = vec_add(elemwise_mul(alpha_l, b), beta_l);
  const Vector term_ln = vec_add(elemwise_mul(alpha_u, b), beta_u);
  const Vector new_lower_p = vec_add(
      vec_add(matvec(lower_M_pos, term_lp), matvec(lower_M_neg, term_ln)), lower_p);

  const Vector term_up = vec_add(elemwise_mul(alpha_u, b), beta_u);
  const Vector term_un = vec_add(elemwise_mul(alpha_l, b), beta_l);
  const Vector new_upper_p = vec_add(
      vec_add(matvec(upper_M_pos, term_up), matvec(upper_M_neg, term_un)), upper_p);

  lower_M = new_lower_M;
  lower_p = new_lower_p;
  upper_M = new_upper_M;
  upper_p = new_upper_p;
}

BackwardBoundResult lirpa_backward_bound(const FullyConnectedNetwork &net,
                     const Vector &x0, double eps,
                     const Matrix *output_lower_M = nullptr,
                     const Vector *output_lower_p = nullptr,
                     const Matrix *output_upper_M = nullptr,
                     const Vector *output_upper_p = nullptr) {
  const ForwardBoundResult fwd = lirpa_forward_bound(net, x0, eps);

  const int output_dim = network_output_dim(net);
  Matrix lower_M = output_lower_M ? *output_lower_M : make_eye(output_dim);
  Matrix upper_M = output_upper_M ? *output_upper_M : make_eye(output_dim);
  Vector lower_p = output_lower_p ? *output_lower_p : make_zero_vector(lower_M.rows);
  Vector upper_p = output_upper_p ? *output_upper_p : make_zero_vector(upper_M.rows);

  require(lower_M.cols == output_dim && upper_M.cols == output_dim,
          "Output spec matrix column mismatch.");
  require(lower_M.rows == upper_M.rows, "Output spec lower/upper row mismatch.");
  require(lower_p.n == lower_M.rows && upper_p.n == upper_M.rows,
          "Output spec vector row mismatch.");

  for (int l = net.num_layers - 1; l >= 0; --l) {
    const LayerBound &lb = fwd.layer_bounds[l];
    backward_one_layer(lower_M, lower_p, upper_M, upper_p, net.W[l], net.b[l],
                       lb.alpha_lower, lb.beta_lower, lb.alpha_upper, lb.beta_upper);
  }

  BackwardBoundResult out;
  out.final_affine.lower_A = lower_M;
  out.final_affine.lower_c = lower_p;
  out.final_affine.upper_A = upper_M;
  out.final_affine.upper_c = upper_p;
  out.final_lower = affine_min(lower_M, lower_p, x0, eps);
  out.final_upper = affine_max(upper_M, upper_p, x0, eps);
  out.num_layer_bounds = fwd.num_layer_bounds;
  for (int i = 0; i < fwd.num_layer_bounds; ++i)
    out.layer_bounds[i] = fwd.layer_bounds[i];
  return out;
}

// GPU 풀 해제 함수
void gpu_pool_cleanup() {
  g_pool.destroy();
}

} // namespace
