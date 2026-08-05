// CPU 전용 CROWN 성능 비교 테스트
// lirpa_forward_backward_fc_array.cpp 기반 (GPU 미사용, 순수 CPU)
// 컴파일: cl /EHsc /utf-8 /F134217728 crown_test_cpu.cpp /Fe:crown_test_cpu.exe
// 또는:   g++ -O0 -Wl,--stack,134217728 -o crown_test_cpu.exe crown_test_cpu.cpp

#include<iostream>
#include<fstream>
#include<vector>
#include<chrono>  // 시간 측정용

// ============================================================
// lirpa_forward_backward_fc_array.cpp 의 내용을 그대로 가져오되
// MAX_DIM 만 512로 변경하여 491차원 신경망을 지원합니다.
// ============================================================

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr int MAX_LAYERS = 16;
constexpr int MAX_DIM = 512;  // 원본은 64 → 491차원 신경망을 위해 512로 확장

enum class ActivationType {
    Relu,
    Sigmoid,
    Linear,
};

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

inline double pos(double x) { return std::max(x, 0.0); }
inline double neg(double x) { return std::min(x, 0.0); }
inline double relu(double x) { return std::max(x, 0.0); }

inline double sigmoid(double x) {
  if (x >= 0.0) {
    return 1.0 / (1.0 + std::exp(-x));
  }
  const double ex = std::exp(x);
  return ex / (1.0 + ex);
}

inline double sigmoid_prime(double x) {
  const double s = sigmoid(x);
  return s * (1.0 - s);
}

void require(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument(msg);
  }
}

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
  for (int i = 0; i < n; ++i) {
    out.a[i][i] = 1.0;
  }
  return out;
}

// === 순수 CPU 연산 함수들 (for 루프) ===

Matrix mat_add(const Matrix &A, const Matrix &B) {
  require(A.rows == B.rows && A.cols == B.cols, "mat_add shape mismatch.");
  Matrix C = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      C.a[i][j] = A.a[i][j] + B.a[i][j];
    }
  }
  return C;
}

Vector vec_add(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_add shape mismatch.");
  Vector out = make_zero_vector(a.n);
  for (int i = 0; i < a.n; ++i) {
    out.v[i] = a.v[i] + b.v[i];
  }
  return out;
}

Vector vec_sub(const Vector &a, const Vector &b) {
  require(a.n == b.n, "vec_sub shape mismatch.");
  Vector out = make_zero_vector(a.n);
  for (int i = 0; i < a.n; ++i) {
    out.v[i] = a.v[i] - b.v[i];
  }
  return out;
}

Matrix matmul(const Matrix &A, const Matrix &B) {
  require(A.cols == B.rows, "matmul shape mismatch.");
  Matrix C = make_zero_matrix(A.rows, B.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int k = 0; k < A.cols; ++k) {
      const double aik = A.a[i][k];
      if (aik == 0.0) {
        continue;
      }
      for (int j = 0; j < B.cols; ++j) {
        C.a[i][j] += aik * B.a[k][j];
      }
    }
  }
  return C;
}

Vector matvec(const Matrix &A, const Vector &x) {
  require(A.cols == x.n, "matvec shape mismatch.");
  Vector y = make_zero_vector(A.rows);
  for (int i = 0; i < A.rows; ++i) {
    double sum = 0.0;
    for (int j = 0; j < A.cols; ++j) {
      sum += A.a[i][j] * x.v[j];
    }
    y.v[i] = sum;
  }
  return y;
}

Matrix positive_part(const Matrix &A) {
  Matrix out = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      out.a[i][j] = pos(A.a[i][j]);
    }
  }
  return out;
}

Matrix negative_part(const Matrix &A) {
  Matrix out = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      out.a[i][j] = neg(A.a[i][j]);
    }
  }
  return out;
}

Matrix rowwise_scale(const Matrix &A, const Vector &s) {
  require(A.rows == s.n, "rowwise_scale shape mismatch.");
  Matrix out = make_zero_matrix(A.rows, A.cols);
  for (int i = 0; i < A.rows; ++i) {
    for (int j = 0; j < A.cols; ++j) {
      out.a[i][j] = A.a[i][j] * s.v[i];
    }
  }
  return out;
}

Vector elemwise_mul(const Vector &a, const Vector &b) {
  require(a.n == b.n, "elemwise_mul shape mismatch.");
  Vector out = make_zero_vector(a.n);
  for (int i = 0; i < a.n; ++i) {
    out.v[i] = a.v[i] * b.v[i];
  }
  return out;
}

Vector make_eps_vec(int n, double eps) {
  Vector out = make_zero_vector(n);
  for (int i = 0; i < n; ++i) {
    out.v[i] = eps;
  }
  return out;
}

Vector affine_min(const Matrix &A, const Vector &c, const Vector &x0,
                  double eps) {
  require(A.rows == c.n && A.cols == x0.n, "affine_min shape mismatch.");
  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);

  Vector out = make_zero_vector(A.rows);
  for (int i = 0; i < A.rows; ++i) {
    double sum = c.v[i];
    for (int j = 0; j < A.cols; ++j) {
      sum += pos(A.a[i][j]) * xl.v[j] + neg(A.a[i][j]) * xu.v[j];
    }
    out.v[i] = sum;
  }
  return out;
}

Vector affine_max(const Matrix &A, const Vector &c, const Vector &x0,
                  double eps) {
  require(A.rows == c.n && A.cols == x0.n, "affine_max shape mismatch.");
  const Vector e = make_eps_vec(x0.n, eps);
  const Vector xl = vec_sub(x0, e);
  const Vector xu = vec_add(x0, e);

  Vector out = make_zero_vector(A.rows);
  for (int i = 0; i < A.rows; ++i) {
    double sum = c.v[i];
    for (int j = 0; j < A.cols; ++j) {
      sum += pos(A.a[i][j]) * xu.v[j] + neg(A.a[i][j]) * xl.v[j];
    }
    out.v[i] = sum;
  }
  return out;
}

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
      // Keep zeros.
    } else {
      const double denom = u - l;
      alpha_u.v[i] = u / denom;
      beta_u.v[i] = -u * l / denom;

      const bool use_identity_lower = std::abs(l) < std::abs(u);
      alpha_l.v[i] = use_identity_lower ? 1.0 : 0.0;
      beta_l.v[i] = 0.0;
    }
  }
}

double bisect_root(double lo, double hi,
                   const std::function<double(double)> &fn, int max_iter = 80,
                   double tol = 1e-12) {
  double flo = fn(lo);
  double fhi = fn(hi);

  if (std::abs(flo) < tol) { return lo; }
  if (std::abs(fhi) < tol) { return hi; }

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
    if (!found) { return xs[best]; }
  }

  for (int it = 0; it < max_iter; ++it) {
    const double mid = 0.5 * (lo + hi);
    const double fmid = fn(mid);
    if (std::abs(fmid) < tol || std::abs(hi - lo) < tol) { return mid; }
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

} // namespace

// ============================================================
// 여기부터 crown_test.cu 와 동일한 main 함수 (모델 로딩 + CROWN 실행)
// ============================================================

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
    for(int i = 0; i < m + 1; ++i) {
        std::cout << sizes[i] << (i == m ? "" : ", ");
    }
    std::cout << std::endl;

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

        std::cout << "- Layer " << i << " Weight: " << n_out << " x " << n_in << std::endl;

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
    
    std::cout << "=== CPU-ONLY CROWN Test ===" << std::endl;
    std::cout << "network read start " << std::endl;
    FullyConnectedNetwork* net = load_custom_network(network_path);
    std::cout << "network read success " << std::endl;

    std::cout << "\ndata read start: " << data_path << std::endl;
    Vector x0 = load_test_data(data_path, net->layer_in_dim[0]);
    std::cout << "data read success (dim=" << x0.n << ")" << std::endl;

    std::cout << "\n========================================\n";
    std::cout << "Starting CROWN Verification (eps = " << eps << ")" << std::endl;
    std::cout << "  ** MODE: CPU ONLY (no GPU) **" << std::endl;
    std::cout << "========================================\n";

    std::cout << std::fixed << std::setprecision(6);

    // 시간 측정 시작
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
                  << std::setw(12) << bwd.final_lower.v[i] << "  |  "
                  << std::setw(12) << bwd.final_upper.v[i] << "  |  "
                  << std::setw(12) << y.v[i] << "\n";
    }
    std::cout << "========================================\n";

    delete net;
    return 0;
}
