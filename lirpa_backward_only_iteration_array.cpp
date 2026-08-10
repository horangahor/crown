#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr int MAX_LAYERS = 16;
constexpr int MAX_DIM = 64;

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
    AffineBound pre_affine;
    Vector pre_lower;
    Vector pre_upper;
    Vector alpha_lower;
    Vector beta_lower;
    Vector alpha_upper;
    Vector beta_upper;
    AffineBound post_affine;
    Vector post_lower;
    Vector post_upper;
};

struct BoundResult {
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

inline void require(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::invalid_argument(msg);
    }
}

inline double pos(double x) {
    return std::max(0.0, x);
}

inline double neg(double x) {
    return std::min(0.0, x);
}

inline double relu(double x) {
    return std::max(0.0, x);
}

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

Matrix make_zero_matrix(int rows, int cols) {
    require(rows >= 0 && rows <= MAX_DIM && cols >= 0 && cols <= MAX_DIM, "Matrix shape out of bounds.");
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

Vector make_eps_vector(int n, double eps) {
    Vector out = make_zero_vector(n);
    for (int i = 0; i < n; ++i) {
        out.v[i] = eps;
    }
    return out;
}

Vector vec_add(const Vector& a, const Vector& b) {
    require(a.n == b.n, "vec_add shape mismatch.");
    Vector out = make_zero_vector(a.n);
    for (int i = 0; i < a.n; ++i) {
        out.v[i] = a.v[i] + b.v[i];
    }
    return out;
}

Vector vec_sub(const Vector& a, const Vector& b) {
    require(a.n == b.n, "vec_sub shape mismatch.");
    Vector out = make_zero_vector(a.n);
    for (int i = 0; i < a.n; ++i) {
        out.v[i] = a.v[i] - b.v[i];
    }
    return out;
}

Vector elemwise_mul(const Vector& a, const Vector& b) {
    require(a.n == b.n, "elemwise_mul shape mismatch.");
    Vector out = make_zero_vector(a.n);
    for (int i = 0; i < a.n; ++i) {
        out.v[i] = a.v[i] * b.v[i];
    }
    return out;
}

Matrix mat_add(const Matrix& A, const Matrix& B) {
    require(A.rows == B.rows && A.cols == B.cols, "mat_add shape mismatch.");
    Matrix C = make_zero_matrix(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            C.a[i][j] = A.a[i][j] + B.a[i][j];
        }
    }
    return C;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
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

Vector matvec(const Matrix& A, const Vector& x) {
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

Matrix positive_part(const Matrix& A) {
    Matrix out = make_zero_matrix(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            out.a[i][j] = pos(A.a[i][j]);
        }
    }
    return out;
}

Matrix negative_part(const Matrix& A) {
    Matrix out = make_zero_matrix(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            out.a[i][j] = neg(A.a[i][j]);
        }
    }
    return out;
}

Vector positive_part(const Vector& x) {
    Vector out = make_zero_vector(x.n);
    for (int i = 0; i < x.n; ++i) {
        out.v[i] = pos(x.v[i]);
    }
    return out;
}

Vector negative_part(const Vector& x) {
    Vector out = make_zero_vector(x.n);
    for (int i = 0; i < x.n; ++i) {
        out.v[i] = neg(x.v[i]);
    }
    return out;
}

Matrix rowwise_scale(const Matrix& A, const Vector& s) {
    require(A.rows == s.n, "rowwise_scale shape mismatch.");
    Matrix out = make_zero_matrix(A.rows, A.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            out.a[i][j] = A.a[i][j] * s.v[i];
        }
    }
    return out;
}

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

Vector affine_min(const Matrix& A, const Vector& c, const Vector& x0, const Vector& eps) {
    require(A.rows == c.n && A.cols == x0.n && x0.n == eps.n, "affine_min shape mismatch.");
    const Vector x_l = vec_sub(x0, eps);
    const Vector x_u = vec_add(x0, eps);

    Vector out = make_zero_vector(A.rows);
    for (int i = 0; i < A.rows; ++i) {
        double sum = c.v[i];
        for (int j = 0; j < A.cols; ++j) {
            sum += pos(A.a[i][j]) * x_l.v[j] + neg(A.a[i][j]) * x_u.v[j];
        }
        out.v[i] = sum;
    }
    return out;
}

Vector affine_max(const Matrix& A, const Vector& c, const Vector& x0, const Vector& eps) {
    require(A.rows == c.n && A.cols == x0.n && x0.n == eps.n, "affine_max shape mismatch.");
    const Vector x_l = vec_sub(x0, eps);
    const Vector x_u = vec_add(x0, eps);

    Vector out = make_zero_vector(A.rows);
    for (int i = 0; i < A.rows; ++i) {
        double sum = c.v[i];
        for (int j = 0; j < A.cols; ++j) {
            sum += pos(A.a[i][j]) * x_u.v[j] + neg(A.a[i][j]) * x_l.v[j];
        }
        out.v[i] = sum;
    }
    return out;
}

struct ReLURelaxation {
    static void relax(
        const Vector& lower,
        const Vector& upper,
        Vector& alpha_l,
        Vector& beta_l,
        Vector& alpha_u,
        Vector& beta_u
    ) {
        require(lower.n == upper.n, "ReLU relax shape mismatch.");
        alpha_l = make_zero_vector(lower.n);
        beta_l = make_zero_vector(lower.n);
        alpha_u = make_zero_vector(lower.n);
        beta_u = make_zero_vector(lower.n);

        for (int i = 0; i < lower.n; ++i) {
            const double l = lower.v[i];
            const double u = upper.v[i];
            require(l <= u, "Invalid interval in ReLU relaxation.");

            if (l >= 0.0) {
                alpha_l.v[i] = 1.0;
                alpha_u.v[i] = 1.0;
            } else if (u <= 0.0) {
                // Keep zero initialization.
            } else {
                const double denom = u - l;
                alpha_u.v[i] = u / denom;
                beta_u.v[i] = -u * l / denom;
                alpha_l.v[i] = (std::abs(l) < std::abs(u)) ? 1.0 : 0.0;
                beta_l.v[i] = 0.0;
            }
        }
    }
};

class SigmoidRelaxation {
public:
    SigmoidRelaxation(int max_iter = 80, double tol = 1e-12)
        : max_iter_(max_iter), tol_(tol) {}

    void relax(
        const Vector& lower,
        const Vector& upper,
        Vector& alpha_l,
        Vector& beta_l,
        Vector& alpha_u,
        Vector& beta_u
    ) const {
        require(lower.n == upper.n, "Sigmoid relax shape mismatch.");
        alpha_l = make_zero_vector(lower.n);
        beta_l = make_zero_vector(lower.n);
        alpha_u = make_zero_vector(lower.n);
        beta_u = make_zero_vector(lower.n);

        for (int i = 0; i < lower.n; ++i) {
            const double l = lower.v[i];
            const double u = upper.v[i];
            require(l <= u, "Invalid interval in Sigmoid relaxation.");

            if (std::abs(u - l) < 1e-14) {
                const double slope = sigmoid_prime(l);
                const double intercept = sigmoid(l) - slope * l;
                alpha_l.v[i] = slope;
                alpha_u.v[i] = slope;
                beta_l.v[i] = intercept;
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
                const double du = crossing_lower_tangent_point(l, u);
                const double dl = crossing_upper_tangent_point(l, u);

                const double slope_lower = sigmoid_prime(du);
                alpha_l.v[i] = slope_lower;
                beta_l.v[i] = sigmoid(du) - slope_lower * du;

                const double slope_upper = sigmoid_prime(dl);
                alpha_u.v[i] = slope_upper;
                beta_u.v[i] = sigmoid(dl) - slope_upper * dl;
            }

            constexpr int samples = 1001;
            double lower_violation = 0.0;
            double upper_violation = 0.0;
            for (int k = 0; k < samples; ++k) {
                const double x = l + (u - l) * static_cast<double>(k) / static_cast<double>(samples - 1);
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

private:
    int max_iter_;
    double tol_;

    double bisect_root(const std::function<double(double)>& fn, double lo, double hi) const {
        double flo = fn(lo);
        double fhi = fn(hi);

        if (std::abs(flo) < tol_) {
            return lo;
        }
        if (std::abs(fhi) < tol_) {
            return hi;
        }

        if (flo * fhi > 0.0) {
            constexpr int N = 257;
            double xs[N]{};
            double vals[N]{};
            for (int i = 0; i < N; ++i) {
                xs[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(N - 1);
                vals[i] = fn(xs[i]);
            }
            int best = 0;
            for (int i = 1; i < N; ++i) {
                if (std::abs(vals[i]) < std::abs(vals[best])) {
                    best = i;
                }
            }

            bool bracketed = false;
            for (int i = 0; i < N - 1; ++i) {
                if (vals[i] == 0.0 || vals[i] * vals[i + 1] <= 0.0) {
                    lo = xs[i];
                    hi = xs[i + 1];
                    flo = vals[i];
                    fhi = vals[i + 1];
                    bracketed = true;
                    break;
                }
            }
            if (!bracketed) {
                return xs[best];
            }
        }

        for (int it = 0; it < max_iter_; ++it) {
            const double mid = 0.5 * (lo + hi);
            const double fmid = fn(mid);
            if (std::abs(fmid) < tol_ || std::abs(hi - lo) < tol_) {
                return mid;
            }
            if (flo * fmid <= 0.0) {
                hi = mid;
                fhi = fmid;
            } else {
                lo = mid;
                flo = fmid;
            }
        }
        return 0.5 * (lo + hi);
    }

    double crossing_lower_tangent_point(double l, double u) const {
        const double su = sigmoid(u);
        auto fn = [&](double d) {
            return (su - sigmoid(d)) / (u - d) - sigmoid_prime(d);
        };
        return bisect_root(fn, l, 0.0);
    }

    double crossing_upper_tangent_point(double l, double u) const {
        const double sl = sigmoid(l);
        auto fn = [&](double d) {
            return (sigmoid(d) - sl) / (d - l) - sigmoid_prime(d);
        };
        return bisect_root(fn, 0.0, u);
    }
};

ActivationType parse_activation(const std::string& name) {
    std::string lowered = name;
    for (char& c : lowered) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    if (lowered == "relu") {
        return ActivationType::Relu;
    }
    if (lowered == "sigmoid") {
        return ActivationType::Sigmoid;
    }
    if (lowered == "linear") {
        return ActivationType::Linear;
    }
    throw std::invalid_argument("Unsupported activation: " + name);
}

FullyConnectedNetwork make_xor_network_from_note() {
    FullyConnectedNetwork net;
    net.num_layers = 2;

    net.layer_in_dim[0] = 2;
    net.layer_out_dim[0] = 2;
    net.W[0] = make_zero_matrix(2, 2);
    net.W[0].a[0][0] = 2.1247;
    net.W[0].a[0][1] = 2.1267;
    net.W[0].a[1][0] = -2.1237;
    net.W[0].a[1][1] = -2.1235;
    net.b[0] = make_zero_vector(2);
    net.b[0].v[0] = -2.1259;
    net.b[0].v[1] = 2.1234;
    net.act[0] = parse_activation("relu");

    net.layer_in_dim[1] = 2;
    net.layer_out_dim[1] = 1;
    net.W[1] = make_zero_matrix(1, 2);
    net.W[1].a[0][0] = -3.6788;
    net.W[1].a[0][1] = -3.6766;
    net.b[1] = make_zero_vector(1);
    net.b[1].v[0] = 3.5451;
    net.act[1] = parse_activation("sigmoid");

    return net;
}

int xor_expected_label(const Vector& x) {
    require(x.n >= 2, "xor_expected_label expects 2D input.");
    return static_cast<int>(std::llround(x.v[0])) ^ static_cast<int>(std::llround(x.v[1]));
}

Vector network_forward(const FullyConnectedNetwork& net, const Vector& x) {
    require(net.num_layers > 0, "Network must have at least one layer.");
    require(x.n == net.layer_in_dim[0], "Input dimension mismatch.");

    Vector f = x;
    for (int l = 0; l < net.num_layers; ++l) {
        const Matrix& W = net.W[l];
        const Vector& b = net.b[l];
        Vector s = vec_add(matvec(W, f), b);

        Vector next = make_zero_vector(s.n);
        for (int i = 0; i < s.n; ++i) {
            if (net.act[l] == ActivationType::Relu) {
                next.v[i] = relu(s.v[i]);
            } else if (net.act[l] == ActivationType::Sigmoid) {
                next.v[i] = sigmoid(s.v[i]);
            } else {
                next.v[i] = s.v[i];
            }
        }
        f = next;
    }
    return f;
}

void print_vector_inline(const Vector& v) {
    std::cout << "[";
    for (int i = 0; i < v.n; ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(6) << v.v[i];
    }
    std::cout << "]";
}

bool allclose(const Vector& a, const Vector& b, double atol, double rtol) {
    if (a.n != b.n) {
        return false;
    }
    for (int i = 0; i < a.n; ++i) {
        const double tol = atol + rtol * std::abs(b.v[i]);
        if (std::abs(a.v[i] - b.v[i]) > tol) {
            return false;
        }
    }
    return true;
}

class LiRPABackwardOnlyReference {
public:
    BoundResult bound(const FullyConnectedNetwork& net, const Vector& x0, const Vector& eps) const {
        validate_input(net, x0, eps);

        BoundResult result;
        result.num_layer_bounds = 0;

        for (int layer_index = 0; layer_index < net.num_layers; ++layer_index) {
            result.layer_bounds[layer_index] = build_one_layer_relaxation(net, layer_index, x0, eps, result.layer_bounds);
            result.num_layer_bounds += 1;
        }

        Matrix lower_M = make_eye(net.layer_out_dim[net.num_layers - 1]);
        Matrix upper_M = make_eye(net.layer_out_dim[net.num_layers - 1]);
        Vector lower_p = make_zero_vector(lower_M.rows);
        Vector upper_p = make_zero_vector(upper_M.rows);

        for (int layer_index = net.num_layers - 1; layer_index >= 0; --layer_index) {
            const LayerBound& lb = result.layer_bounds[layer_index];
            backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                net.W[layer_index], net.b[layer_index],
                lb.alpha_lower, lb.beta_lower,
                lb.alpha_upper, lb.beta_upper,
                lower_M, lower_p, upper_M, upper_p
            );
        }

        result.final_affine = AffineBound{lower_M, lower_p, upper_M, upper_p};
        result.final_lower = affine_min(result.final_affine.lower_A, result.final_affine.lower_c, x0, eps);
        result.final_upper = affine_max(result.final_affine.upper_A, result.final_affine.upper_c, x0, eps);
        return result;
    }

private:
    static void validate_input(const FullyConnectedNetwork& net, const Vector& x0, const Vector& eps) {
        require(net.num_layers > 0, "Network must have at least one layer.");
        require(x0.n == net.layer_in_dim[0], "x0 dimension mismatch.");
        require(eps.n == x0.n, "eps must match x0 dimension.");
        for (int i = 0; i < eps.n; ++i) {
            require(eps.v[i] >= 0.0, "eps must be nonnegative.");
        }
    }

    static void linear_relaxation(const Vector& lower, Vector& alpha_l, Vector& beta_l, Vector& alpha_u, Vector& beta_u) {
        alpha_l = make_zero_vector(lower.n);
        beta_l = make_zero_vector(lower.n);
        alpha_u = make_zero_vector(lower.n);
        beta_u = make_zero_vector(lower.n);
        for (int i = 0; i < lower.n; ++i) {
            alpha_l.v[i] = 1.0;
            alpha_u.v[i] = 1.0;
        }
    }

    static void backward_one_layer(
        const Matrix& lower_M,
        const Vector& lower_p,
        const Matrix& upper_M,
        const Vector& upper_p,
        const Matrix& W,
        const Vector& b,
        const Vector& alpha_l,
        const Vector& beta_l,
        const Vector& alpha_u,
        const Vector& beta_u,
        Matrix& out_lower_M,
        Vector& out_lower_p,
        Matrix& out_upper_M,
        Vector& out_upper_p
    ) {
        const Matrix lower_M_pos = positive_part(lower_M);
        const Matrix lower_M_neg = negative_part(lower_M);
        const Matrix upper_M_pos = positive_part(upper_M);
        const Matrix upper_M_neg = negative_part(upper_M);

        const Matrix lower_s_coeff = mat_add(colwise_scale(lower_M_pos, alpha_l), colwise_scale(lower_M_neg, alpha_u));
        out_lower_M = matmul(lower_s_coeff, W);

        const Vector alpha_lb = vec_add(elemwise_mul(alpha_l, b), beta_l);
        const Vector alpha_ub = vec_add(elemwise_mul(alpha_u, b), beta_u);
        out_lower_p = vec_add(vec_add(matvec(lower_M_pos, alpha_lb), matvec(lower_M_neg, alpha_ub)), lower_p);

        const Matrix upper_s_coeff = mat_add(colwise_scale(upper_M_pos, alpha_u), colwise_scale(upper_M_neg, alpha_l));
        out_upper_M = matmul(upper_s_coeff, W);
        out_upper_p = vec_add(vec_add(matvec(upper_M_pos, alpha_ub), matvec(upper_M_neg, alpha_lb)), upper_p);
    }

    static AffineBound compose_post_activation_bound(
        const AffineBound& pre,
        const Vector& alpha_l,
        const Vector& beta_l,
        const Vector& alpha_u,
        const Vector& beta_u
    ) {
        const Vector alpha_l_pos = positive_part(alpha_l);
        const Vector alpha_l_neg = negative_part(alpha_l);
        const Vector alpha_u_pos = positive_part(alpha_u);
        const Vector alpha_u_neg = negative_part(alpha_u);

        const Matrix lower_A = mat_add(rowwise_scale(pre.lower_A, alpha_l_pos), rowwise_scale(pre.upper_A, alpha_l_neg));
        const Matrix upper_A = mat_add(rowwise_scale(pre.upper_A, alpha_u_pos), rowwise_scale(pre.lower_A, alpha_u_neg));

        const Vector lower_c = vec_add(vec_add(elemwise_mul(alpha_l_pos, pre.lower_c), elemwise_mul(alpha_l_neg, pre.upper_c)), beta_l);
        const Vector upper_c = vec_add(vec_add(elemwise_mul(alpha_u_pos, pre.upper_c), elemwise_mul(alpha_u_neg, pre.lower_c)), beta_u);

        return AffineBound{lower_A, lower_c, upper_A, upper_c};
    }

    static LayerBound build_one_layer_relaxation(
        const FullyConnectedNetwork& net,
        int layer_index,
        const Vector& x0,
        const Vector& eps,
        const LayerBound* previous_layer_bounds
    ) {
        Matrix lower_M = net.W[layer_index];
        Matrix upper_M = net.W[layer_index];
        Vector lower_p = net.b[layer_index];
        Vector upper_p = net.b[layer_index];

        for (int prev = layer_index - 1; prev >= 0; --prev) {
            const LayerBound& lb = previous_layer_bounds[prev];
            backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                net.W[prev], net.b[prev],
                lb.alpha_lower, lb.beta_lower,
                lb.alpha_upper, lb.beta_upper,
                lower_M, lower_p, upper_M, upper_p
            );
        }

        AffineBound pre{lower_M, lower_p, upper_M, upper_p};
        Vector pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
        Vector pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

        Vector alpha_l, beta_l, alpha_u, beta_u;
        if (net.act[layer_index] == ActivationType::Linear) {
            linear_relaxation(pre_lower, alpha_l, beta_l, alpha_u, beta_u);
        } else if (net.act[layer_index] == ActivationType::Relu) {
            ReLURelaxation::relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
        } else {
            SigmoidRelaxation().relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
        }

        const AffineBound post = compose_post_activation_bound(pre, alpha_l, beta_l, alpha_u, beta_u);
        const Vector post_lower = affine_min(post.lower_A, post.lower_c, x0, eps);
        const Vector post_upper = affine_max(post.upper_A, post.upper_c, x0, eps);

        return LayerBound{pre, pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u, post, post_lower, post_upper};
    }
};

class LiRPABackwardOnlyIterationArray {
public:
    BoundResult bound(const FullyConnectedNetwork& net, const Vector& x0, const Vector& eps, bool debug = false) const {
        validate_input(net, x0, eps);

        BoundResult result;
        result.num_layer_bounds = 0;
        build_layer_relaxations_iterative(net, x0, eps, debug, result.layer_bounds, result.num_layer_bounds);

        Matrix lower_M = make_eye(net.layer_out_dim[net.num_layers - 1]);
        Matrix upper_M = make_eye(net.layer_out_dim[net.num_layers - 1]);
        Vector lower_p = make_zero_vector(lower_M.rows);
        Vector upper_p = make_zero_vector(upper_M.rows);

        for (int layer_index = net.num_layers - 1; layer_index >= 0; --layer_index) {
            const LayerBound& lb = result.layer_bounds[layer_index];
            backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                net.W[layer_index], net.b[layer_index],
                lb.alpha_lower, lb.beta_lower,
                lb.alpha_upper, lb.beta_upper,
                lower_M, lower_p, upper_M, upper_p
            );
        }

        result.final_affine = AffineBound{lower_M, lower_p, upper_M, upper_p};
        result.final_lower = affine_min(result.final_affine.lower_A, result.final_affine.lower_c, x0, eps);
        result.final_upper = affine_max(result.final_affine.upper_A, result.final_affine.upper_c, x0, eps);
        return result;
    }

private:
    static void validate_input(const FullyConnectedNetwork& net, const Vector& x0, const Vector& eps) {
        require(net.num_layers > 0, "Network must have at least one layer.");
        require(x0.n == net.layer_in_dim[0], "x0 dimension mismatch.");
        require(eps.n == x0.n, "eps must match x0 dimension.");
        for (int i = 0; i < eps.n; ++i) {
            require(eps.v[i] >= 0.0, "eps must be nonnegative.");
        }
    }

    static void linear_relaxation(const Vector& lower, Vector& alpha_l, Vector& beta_l, Vector& alpha_u, Vector& beta_u) {
        alpha_l = make_zero_vector(lower.n);
        beta_l = make_zero_vector(lower.n);
        alpha_u = make_zero_vector(lower.n);
        beta_u = make_zero_vector(lower.n);
        for (int i = 0; i < lower.n; ++i) {
            alpha_l.v[i] = 1.0;
            alpha_u.v[i] = 1.0;
        }
    }

    static void backward_one_layer(
        const Matrix& lower_M,
        const Vector& lower_p,
        const Matrix& upper_M,
        const Vector& upper_p,
        const Matrix& W,
        const Vector& b,
        const Vector& alpha_l,
        const Vector& beta_l,
        const Vector& alpha_u,
        const Vector& beta_u,
        Matrix& out_lower_M,
        Vector& out_lower_p,
        Matrix& out_upper_M,
        Vector& out_upper_p
    ) {
        const Matrix lower_M_pos = positive_part(lower_M);
        const Matrix lower_M_neg = negative_part(lower_M);
        const Matrix upper_M_pos = positive_part(upper_M);
        const Matrix upper_M_neg = negative_part(upper_M);

        const Matrix lower_s_coeff = mat_add(colwise_scale(lower_M_pos, alpha_l), colwise_scale(lower_M_neg, alpha_u));
        out_lower_M = matmul(lower_s_coeff, W);

        const Vector alpha_lb = vec_add(elemwise_mul(alpha_l, b), beta_l);
        const Vector alpha_ub = vec_add(elemwise_mul(alpha_u, b), beta_u);
        out_lower_p = vec_add(vec_add(matvec(lower_M_pos, alpha_lb), matvec(lower_M_neg, alpha_ub)), lower_p);

        const Matrix upper_s_coeff = mat_add(colwise_scale(upper_M_pos, alpha_u), colwise_scale(upper_M_neg, alpha_l));
        out_upper_M = matmul(upper_s_coeff, W);
        out_upper_p = vec_add(vec_add(matvec(upper_M_pos, alpha_ub), matvec(upper_M_neg, alpha_lb)), upper_p);
    }

    static AffineBound compose_post_activation_bound(
        const AffineBound& pre,
        const Vector& alpha_l,
        const Vector& beta_l,
        const Vector& alpha_u,
        const Vector& beta_u
    ) {
        const Vector alpha_l_pos = positive_part(alpha_l);
        const Vector alpha_l_neg = negative_part(alpha_l);
        const Vector alpha_u_pos = positive_part(alpha_u);
        const Vector alpha_u_neg = negative_part(alpha_u);

        const Matrix lower_A = mat_add(rowwise_scale(pre.lower_A, alpha_l_pos), rowwise_scale(pre.upper_A, alpha_l_neg));
        const Matrix upper_A = mat_add(rowwise_scale(pre.upper_A, alpha_u_pos), rowwise_scale(pre.lower_A, alpha_u_neg));

        const Vector lower_c = vec_add(vec_add(elemwise_mul(alpha_l_pos, pre.lower_c), elemwise_mul(alpha_l_neg, pre.upper_c)), beta_l);
        const Vector upper_c = vec_add(vec_add(elemwise_mul(alpha_u_pos, pre.upper_c), elemwise_mul(alpha_u_neg, pre.lower_c)), beta_u);

        return AffineBound{lower_A, lower_c, upper_A, upper_c};
    }

    static void print_layer_debug_info(int layer_index, const LayerBound& lb) {
        std::cout << "[debug] layer " << (layer_index + 1) << "\n";
        std::cout << "  pre_lower=";
        print_vector_inline(lb.pre_lower);
        std::cout << "\n  pre_upper=";
        print_vector_inline(lb.pre_upper);
        std::cout << "\n  alpha_lower=";
        print_vector_inline(lb.alpha_lower);
        std::cout << "\n  beta_lower=";
        print_vector_inline(lb.beta_lower);
        std::cout << "\n  alpha_upper=";
        print_vector_inline(lb.alpha_upper);
        std::cout << "\n  beta_upper=";
        print_vector_inline(lb.beta_upper);
        std::cout << "\n";
    }

    static void build_layer_relaxations_iterative(
        const FullyConnectedNetwork& net,
        const Vector& x0,
        const Vector& eps,
        bool debug,
        LayerBound* layer_bounds,
        int& num_layer_bounds
    ) {
        num_layer_bounds = 0;

        for (int k = 0; k < net.num_layers; ++k) {
            Matrix lower_C = net.W[k];
            Matrix upper_C = net.W[k];
            Vector lower_d = net.b[k];
            Vector upper_d = net.b[k];

            for (int r = k - 1; r >= 0; --r) {
                const LayerBound& prev = layer_bounds[r];

                const Matrix lower_C_pos = positive_part(lower_C);
                const Matrix lower_C_neg = negative_part(lower_C);
                lower_C = matmul(
                    mat_add(colwise_scale(lower_C_pos, prev.alpha_lower), colwise_scale(lower_C_neg, prev.alpha_upper)),
                    net.W[r]
                );
                lower_d = vec_add(
                    lower_d,
                    vec_add(
                        matvec(lower_C_pos, vec_add(elemwise_mul(prev.alpha_lower, net.b[r]), prev.beta_lower)),
                        matvec(lower_C_neg, vec_add(elemwise_mul(prev.alpha_upper, net.b[r]), prev.beta_upper))
                    )
                );

                const Matrix upper_C_pos = positive_part(upper_C);
                const Matrix upper_C_neg = negative_part(upper_C);
                upper_C = matmul(
                    mat_add(colwise_scale(upper_C_pos, prev.alpha_upper), colwise_scale(upper_C_neg, prev.alpha_lower)),
                    net.W[r]
                );
                upper_d = vec_add(
                    upper_d,
                    vec_add(
                        matvec(upper_C_pos, vec_add(elemwise_mul(prev.alpha_upper, net.b[r]), prev.beta_upper)),
                        matvec(upper_C_neg, vec_add(elemwise_mul(prev.alpha_lower, net.b[r]), prev.beta_lower))
                    )
                );
            }

            const AffineBound pre{lower_C, lower_d, upper_C, upper_d};
            const Vector pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
            const Vector pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

            Vector alpha_l, beta_l, alpha_u, beta_u;
            if (net.act[k] == ActivationType::Linear) {
                linear_relaxation(pre_lower, alpha_l, beta_l, alpha_u, beta_u);
            } else if (net.act[k] == ActivationType::Relu) {
                ReLURelaxation::relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
            } else {
                SigmoidRelaxation().relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
            }

            const AffineBound post = compose_post_activation_bound(pre, alpha_l, beta_l, alpha_u, beta_u);
            const Vector post_lower = affine_min(post.lower_A, post.lower_c, x0, eps);
            const Vector post_upper = affine_max(post.upper_A, post.upper_c, x0, eps);

            layer_bounds[k] = LayerBound{pre, pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u, post, post_lower, post_upper};
            num_layer_bounds += 1;

            if (debug) {
                print_layer_debug_info(k, layer_bounds[k]);
            }
        }
    }
};

Vector make_point(double x0, double x1) {
    Vector x = make_zero_vector(2);
    x.v[0] = x0;
    x.v[1] = x1;
    return x;
}

void compare_reference_and_iterative(double eps_scalar = 0.02, double atol = 1e-9, double rtol = 1e-9) {
    const FullyConnectedNetwork net = make_xor_network_from_note();
    LiRPABackwardOnlyReference reference;
    LiRPABackwardOnlyIterationArray iterative;

    const std::array<Vector, 4> points = {
        make_point(0.0, 0.0),
        make_point(0.0, 1.0),
        make_point(1.0, 0.0),
        make_point(1.0, 1.0),
    };

    std::cout << "Comparing reference vs iterative backward-only bounds (array)\n";
    std::cout << "Tolerance: atol=" << atol << ", rtol=" << rtol << "\n\n";

    for (const auto& x0 : points) {
        const Vector eps = make_eps_vector(x0.n, eps_scalar);

        const BoundResult ref_res = reference.bound(net, x0, eps);
        const BoundResult itr_res = iterative.bound(net, x0, eps, false);

        require(allclose(ref_res.final_lower, itr_res.final_lower, atol, rtol), "Lower bound mismatch.");
        require(allclose(ref_res.final_upper, itr_res.final_upper, atol, rtol), "Upper bound mismatch.");
        require(ref_res.num_layer_bounds == itr_res.num_layer_bounds, "Layer count mismatch.");

        for (int k = 0; k < ref_res.num_layer_bounds; ++k) {
            require(allclose(ref_res.layer_bounds[k].pre_lower, itr_res.layer_bounds[k].pre_lower, atol, rtol), "pre_lower mismatch.");
            require(allclose(ref_res.layer_bounds[k].pre_upper, itr_res.layer_bounds[k].pre_upper, atol, rtol), "pre_upper mismatch.");
            require(allclose(ref_res.layer_bounds[k].alpha_lower, itr_res.layer_bounds[k].alpha_lower, atol, rtol), "alpha_lower mismatch.");
            require(allclose(ref_res.layer_bounds[k].beta_lower, itr_res.layer_bounds[k].beta_lower, atol, rtol), "beta_lower mismatch.");
            require(allclose(ref_res.layer_bounds[k].alpha_upper, itr_res.layer_bounds[k].alpha_upper, atol, rtol), "alpha_upper mismatch.");
            require(allclose(ref_res.layer_bounds[k].beta_upper, itr_res.layer_bounds[k].beta_upper, atol, rtol), "beta_upper mismatch.");
        }

        std::cout << "x0=";
        print_vector_inline(x0);
        std::cout << ": OK\n";
    }

    std::cout << "\nReference and iterative backward-only array results match for all tested points.\n";
}

void run_xor_demo_iterative_array(double eps_scalar = 0.02, bool debug = false) {
    const FullyConnectedNetwork net = make_xor_network_from_note();
    LiRPABackwardOnlyIterationArray verifier;

    const std::array<Vector, 4> points = {
        make_point(0.0, 0.0),
        make_point(0.0, 1.0),
        make_point(1.0, 0.0),
        make_point(1.0, 1.0),
    };

    std::cout << "XOR iterative backward-only CROWN bounds (array)\n";
    std::cout << "Perturbation: L_inf epsilon = " << eps_scalar << "\n\n";

    bool all_certified = true;
    for (const auto& x0 : points) {
        const Vector eps = make_eps_vector(x0.n, eps_scalar);
        const Vector y = network_forward(net, x0);
        const BoundResult res = verifier.bound(net, x0, eps, debug);

        const int expected = xor_expected_label(x0);
        bool certified = false;
        std::string condition;
        if (expected == 1) {
            certified = res.final_lower.v[0] > 0.5;
            condition = "lower bound > 0.5";
        } else {
            certified = res.final_upper.v[0] < 0.5;
            condition = "upper bound < 0.5";
        }
        all_certified = all_certified && certified;

        std::cout << std::fixed << std::setprecision(6)
                  << "x0=";
        print_vector_inline(x0);
        std::cout << ", expected=" << expected << ", network_output=" << y.v[0] << "\n";
        std::cout << "  iterative backward-only bound=[" << res.final_lower.v[0] << ", " << res.final_upper.v[0] << "]"
                  << ", certified=" << (certified ? "True" : "False")
                  << " (" << condition << ")\n\n";
    }

    if (all_certified) {
        std::cout << "Iterative backward-only array mode certifies all four XOR corner classifications for this epsilon.\n";
    } else {
        std::cout << "Iterative backward-only array mode does not certify at least one XOR corner classification for this epsilon.\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        double eps = 0.02;
        bool debug = false;
        bool do_compare = true;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--debug") {
                debug = true;
            } else if (arg == "--no-compare") {
                do_compare = false;
            } else {
                eps = std::stod(arg);
            }
        }

        if (do_compare) {
            compare_reference_and_iterative(eps);
            std::cout << "\n";
        }
        run_xor_demo_iterative_array(eps, debug);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
