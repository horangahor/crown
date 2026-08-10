#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

struct AffineBound {
    Mat lower_A;
    Vec lower_c;
    Mat upper_A;
    Vec upper_c;
};

struct LayerBound {
    AffineBound pre_affine;
    Vec pre_lower;
    Vec pre_upper;
    Vec alpha_lower;
    Vec beta_lower;
    Vec alpha_upper;
    Vec beta_upper;
    AffineBound post_affine;
    Vec post_lower;
    Vec post_upper;
};

static Mat zeros_mat(int rows, int cols) {
    return Mat(rows, Vec(cols, 0.0));
}

static Mat eye(int dim) {
    Mat out = zeros_mat(dim, dim);
    for (int i = 0; i < dim; ++i) {
        out[i][i] = 1.0;
    }
    return out;
}

static void check_matrix(const Mat& A, const std::string& name) {
    if (A.empty()) {
        return;
    }
    const std::size_t cols = A[0].size();
    for (const auto& row : A) {
        if (row.size() != cols) {
            throw std::invalid_argument(name + " has ragged rows.");
        }
    }
}

static Mat matmul(const Mat& A, const Mat& B) {
    check_matrix(A, "A");
    check_matrix(B, "B");
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("matmul requires non-empty matrices.");
    }
    const int m = static_cast<int>(A.size());
    const int k = static_cast<int>(A[0].size());
    const int k2 = static_cast<int>(B.size());
    const int n = static_cast<int>(B[0].size());
    if (k != k2) {
        throw std::invalid_argument("matmul shape mismatch.");
    }

    Mat C = zeros_mat(m, n);
    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < k; ++p) {
            const double a = A[i][p];
            if (a == 0.0) {
                continue;
            }
            for (int j = 0; j < n; ++j) {
                C[i][j] += a * B[p][j];
            }
        }
    }
    return C;
}

static Vec matvec(const Mat& A, const Vec& x) {
    check_matrix(A, "A");
    if (A.empty()) {
        return {};
    }
    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    if (static_cast<int>(x.size()) != n) {
        throw std::invalid_argument("matvec shape mismatch.");
    }

    Vec y(m, 0.0);
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

static Mat mat_add(const Mat& A, const Mat& B) {
    if (A.size() != B.size() || (!A.empty() && A[0].size() != B[0].size())) {
        throw std::invalid_argument("mat_add shape mismatch.");
    }
    Mat out = A;
    for (std::size_t i = 0; i < out.size(); ++i) {
        for (std::size_t j = 0; j < out[i].size(); ++j) {
            out[i][j] += B[i][j];
        }
    }
    return out;
}

static Vec vec_add(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vec_add shape mismatch.");
    }
    Vec out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

static Vec vec_sub(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vec_sub shape mismatch.");
    }
    Vec out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

static Vec vec_mul(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vec_mul shape mismatch.");
    }
    Vec out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

static Vec vec_scale(const Vec& a, double s) {
    Vec out(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * s;
    }
    return out;
}

static Mat positive_part(const Mat& A) {
    Mat out = A;
    for (auto& row : out) {
        for (double& x : row) {
            x = std::max(0.0, x);
        }
    }
    return out;
}

static Mat negative_part(const Mat& A) {
    Mat out = A;
    for (auto& row : out) {
        for (double& x : row) {
            x = std::min(0.0, x);
        }
    }
    return out;
}

static Vec positive_part(const Vec& v) {
    Vec out = v;
    for (double& x : out) {
        x = std::max(0.0, x);
    }
    return out;
}

static Vec negative_part(const Vec& v) {
    Vec out = v;
    for (double& x : out) {
        x = std::min(0.0, x);
    }
    return out;
}

static Mat rowwise_scale(const Mat& A, const Vec& s) {
    if (A.size() != s.size()) {
        throw std::invalid_argument("rowwise_scale shape mismatch.");
    }
    Mat out = A;
    for (std::size_t i = 0; i < out.size(); ++i) {
        for (double& x : out[i]) {
            x *= s[i];
        }
    }
    return out;
}

static Mat colwise_scale(const Mat& A, const Vec& s) {
    if (A.empty()) {
        return A;
    }
    if (A[0].size() != s.size()) {
        throw std::invalid_argument("colwise_scale shape mismatch.");
    }
    Mat out = A;
    for (std::size_t i = 0; i < out.size(); ++i) {
        for (std::size_t j = 0; j < out[i].size(); ++j) {
            out[i][j] *= s[j];
        }
    }
    return out;
}

static Vec affine_min(const Mat& A, const Vec& c, const Vec& x0, const Vec& eps) {
    if (x0.size() != eps.size()) {
        throw std::invalid_argument("affine_min eps size mismatch.");
    }
    Vec x_l = vec_sub(x0, eps);
    Vec x_u = vec_add(x0, eps);

    if (A.empty()) {
        return c;
    }

    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    if (static_cast<int>(c.size()) != m || static_cast<int>(x_l.size()) != n) {
        throw std::invalid_argument("affine_min shape mismatch.");
    }

    Vec out(m, 0.0);
    for (int i = 0; i < m; ++i) {
        double sum = c[i];
        for (int j = 0; j < n; ++j) {
            sum += std::max(A[i][j], 0.0) * x_l[j] + std::min(A[i][j], 0.0) * x_u[j];
        }
        out[i] = sum;
    }
    return out;
}

static Vec affine_max(const Mat& A, const Vec& c, const Vec& x0, const Vec& eps) {
    if (x0.size() != eps.size()) {
        throw std::invalid_argument("affine_max eps size mismatch.");
    }
    Vec x_l = vec_sub(x0, eps);
    Vec x_u = vec_add(x0, eps);

    if (A.empty()) {
        return c;
    }

    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    if (static_cast<int>(c.size()) != m || static_cast<int>(x_u.size()) != n) {
        throw std::invalid_argument("affine_max shape mismatch.");
    }

    Vec out(m, 0.0);
    for (int i = 0; i < m; ++i) {
        double sum = c[i];
        for (int j = 0; j < n; ++j) {
            sum += std::max(A[i][j], 0.0) * x_u[j] + std::min(A[i][j], 0.0) * x_l[j];
        }
        out[i] = sum;
    }
    return out;
}

static Vec ones_like(const Vec& x) {
    return Vec(x.size(), 1.0);
}

class ReLURelaxation {
public:
    static double relu(double x) {
        return std::max(0.0, x);
    }

    static void relax(const Vec& lower, const Vec& upper, Vec& alpha_l, Vec& beta_l, Vec& alpha_u, Vec& beta_u) {
        if (lower.size() != upper.size()) {
            throw std::invalid_argument("ReLU relax size mismatch.");
        }

        const int n = static_cast<int>(lower.size());
        alpha_l.assign(n, 0.0);
        beta_l.assign(n, 0.0);
        alpha_u.assign(n, 0.0);
        beta_u.assign(n, 0.0);

        for (int i = 0; i < n; ++i) {
            const double l = lower[i];
            const double u = upper[i];
            if (l > u) {
                throw std::invalid_argument("Invalid interval in ReLU relaxation.");
            }

            if (l >= 0.0) {
                alpha_l[i] = 1.0;
                alpha_u[i] = 1.0;
            } else if (u <= 0.0) {
                // Keep zeros.
            } else {
                const double denom = u - l;
                alpha_u[i] = u / denom;
                beta_u[i] = -u * l / denom;
                alpha_l[i] = (std::abs(l) < std::abs(u)) ? 1.0 : 0.0;
                beta_l[i] = 0.0;
            }
        }
    }
};

class SigmoidRelaxation {
public:
    SigmoidRelaxation(int max_iter = 80, double tol = 1e-12)
        : max_iter_(max_iter), tol_(tol) {}

    static double sigma(double x) {
        if (x >= 0.0) {
            return 1.0 / (1.0 + std::exp(-x));
        }
        const double ex = std::exp(x);
        return ex / (1.0 + ex);
    }

    static double sigma_prime(double x) {
        const double s = sigma(x);
        return s * (1.0 - s);
    }

    void relax(const Vec& lower, const Vec& upper, Vec& alpha_l, Vec& beta_l, Vec& alpha_u, Vec& beta_u) const {
        if (lower.size() != upper.size()) {
            throw std::invalid_argument("Sigmoid relax size mismatch.");
        }

        const int n = static_cast<int>(lower.size());
        alpha_l.assign(n, 0.0);
        beta_l.assign(n, 0.0);
        alpha_u.assign(n, 0.0);
        beta_u.assign(n, 0.0);

        for (int i = 0; i < n; ++i) {
            const double l = lower[i];
            const double u = upper[i];
            if (l > u) {
                throw std::invalid_argument("Invalid interval in Sigmoid relaxation.");
            }

            if (std::abs(u - l) < 1e-14) {
                const double slope = sigma_prime(l);
                const double intercept = sigma(l) - slope * l;
                alpha_l[i] = slope;
                alpha_u[i] = slope;
                beta_l[i] = intercept;
                beta_u[i] = intercept;
                continue;
            }

            if (l >= 0.0) {
                const double slope_sec = (sigma(u) - sigma(l)) / (u - l);
                alpha_l[i] = slope_sec;
                beta_l[i] = sigma(u) - slope_sec * u;

                const double x0 = 0.5 * (l + u);
                const double slope_tan = sigma_prime(x0);
                alpha_u[i] = slope_tan;
                beta_u[i] = sigma(x0) - slope_tan * x0;
            } else if (u <= 0.0) {
                const double x0 = 0.5 * (l + u);
                const double slope_tan = sigma_prime(x0);
                alpha_l[i] = slope_tan;
                beta_l[i] = sigma(x0) - slope_tan * x0;

                const double slope_sec = (sigma(u) - sigma(l)) / (u - l);
                alpha_u[i] = slope_sec;
                beta_u[i] = sigma(u) - slope_sec * u;
            } else {
                const double du = crossing_lower_tangent_point(l, u);
                const double dl = crossing_upper_tangent_point(l, u);

                const double slope_lower = sigma_prime(du);
                alpha_l[i] = slope_lower;
                beta_l[i] = sigma(du) - slope_lower * du;

                const double slope_upper = sigma_prime(dl);
                alpha_u[i] = slope_upper;
                beta_u[i] = sigma(dl) - slope_upper * dl;
            }

            double lower_violation = 0.0;
            double upper_violation = 0.0;
            constexpr int samples = 1001;
            for (int k = 0; k < samples; ++k) {
                const double x = l + (u - l) * static_cast<double>(k) / static_cast<double>(samples - 1);
                const double y = sigma(x);
                const double lower_line = alpha_l[i] * x + beta_l[i];
                const double upper_line = alpha_u[i] * x + beta_u[i];
                lower_violation = std::max(lower_violation, lower_line - y);
                upper_violation = std::max(upper_violation, y - upper_line);
            }
            if (lower_violation > 1e-10) {
                beta_l[i] -= lower_violation + 1e-10;
            }
            if (upper_violation > 1e-10) {
                beta_u[i] += upper_violation + 1e-10;
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
            constexpr int grid_n = 257;
            std::vector<double> xs(grid_n);
            std::vector<double> vals(grid_n);
            for (int i = 0; i < grid_n; ++i) {
                xs[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(grid_n - 1);
                vals[i] = fn(xs[i]);
            }

            int best = 0;
            for (int i = 1; i < grid_n; ++i) {
                if (std::abs(vals[i]) < std::abs(vals[best])) {
                    best = i;
                }
            }

            bool bracketed = false;
            for (int i = 0; i < grid_n - 1; ++i) {
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
        const double su = sigma(u);
        auto fn = [&](double d) {
            return (su - sigma(d)) / (u - d) - sigma_prime(d);
        };
        return bisect_root(fn, l, 0.0);
    }

    double crossing_upper_tangent_point(double l, double u) const {
        const double sl = sigma(l);
        auto fn = [&](double d) {
            return (sigma(d) - sl) / (d - l) - sigma_prime(d);
        };
        return bisect_root(fn, 0.0, u);
    }
};

class FullyConnectedNetwork {
public:
    FullyConnectedNetwork(std::vector<Mat> weights, std::vector<Vec> biases, std::vector<std::string> activations)
        : weights_(std::move(weights)), biases_(std::move(biases)), activations_(std::move(activations)) {
        if (!(weights_.size() == biases_.size() && biases_.size() == activations_.size())) {
            throw std::invalid_argument("weights, biases, and activations must have the same length.");
        }
        if (weights_.empty()) {
            throw std::invalid_argument("Network must have at least one layer.");
        }
        for (std::size_t i = 0; i < activations_.size(); ++i) {
            for (char& ch : activations_[i]) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
        }

        for (std::size_t layer = 0; layer < weights_.size(); ++layer) {
            check_matrix(weights_[layer], "W");
            if (weights_[layer].empty()) {
                throw std::invalid_argument("Weight matrices must be non-empty.");
            }
            if (weights_[layer].size() != biases_[layer].size()) {
                throw std::invalid_argument("W rows must match b length.");
            }
            if (layer > 0) {
                if (weights_[layer - 1].size() != weights_[layer][0].size()) {
                    throw std::invalid_argument("Layer dimension mismatch.");
                }
            }
        }
    }

    int input_dim() const {
        return static_cast<int>(weights_[0][0].size());
    }

    const std::vector<Mat>& weights() const { return weights_; }
    const std::vector<Vec>& biases() const { return biases_; }
    const std::vector<std::string>& activations() const { return activations_; }

    Vec forward(const Vec& x) const {
        Vec f = x;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            Vec s = vec_add(matvec(weights_[i], f), biases_[i]);
            const std::string& act = activations_[i];
            if (act == "relu") {
                for (double& v : s) {
                    v = ReLURelaxation::relu(v);
                }
                f = s;
            } else if (act == "sigmoid") {
                for (double& v : s) {
                    v = SigmoidRelaxation::sigma(v);
                }
                f = s;
            } else if (act == "linear") {
                f = s;
            } else {
                throw std::invalid_argument("Unsupported activation: " + act);
            }
        }
        return f;
    }

private:
    std::vector<Mat> weights_;
    std::vector<Vec> biases_;
    std::vector<std::string> activations_;
};

class LiRPABackwardOnlyReference {
public:
    std::tuple<AffineBound, Vec, Vec, std::vector<LayerBound>> bound(
        const FullyConnectedNetwork& network,
        const Vec& x0,
        const Vec& eps
    ) const {
        validate_input(network, x0, eps);

        std::vector<LayerBound> layer_bounds;
        const int L = static_cast<int>(network.weights().size());
        for (int layer_index = 0; layer_index < L; ++layer_index) {
            layer_bounds.push_back(build_one_layer_relaxation(network, layer_index, x0, eps, layer_bounds));
        }

        const int output_dim = static_cast<int>(network.weights().back().size());
        Mat lower_M = eye(output_dim);
        Mat upper_M = eye(output_dim);
        Vec lower_p(output_dim, 0.0);
        Vec upper_p(output_dim, 0.0);

        for (int layer_index = L - 1; layer_index >= 0; --layer_index) {
            const LayerBound& lb = layer_bounds[layer_index];
            std::tie(lower_M, lower_p, upper_M, upper_p) = backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                network.weights()[layer_index], network.biases()[layer_index],
                lb.alpha_lower, lb.beta_lower, lb.alpha_upper, lb.beta_upper
            );
        }

        AffineBound final{lower_M, lower_p, upper_M, upper_p};
        Vec final_lower = affine_min(final.lower_A, final.lower_c, x0, eps);
        Vec final_upper = affine_max(final.upper_A, final.upper_c, x0, eps);
        return {final, final_lower, final_upper, layer_bounds};
    }

private:
    static void linear_relaxation(const Vec& lower, Vec& alpha_l, Vec& beta_l, Vec& alpha_u, Vec& beta_u) {
        alpha_l = ones_like(lower);
        beta_l = Vec(lower.size(), 0.0);
        alpha_u = alpha_l;
        beta_u = beta_l;
    }

    static void validate_input(const FullyConnectedNetwork& network, const Vec& x0, const Vec& eps) {
        if (x0.empty()) {
            throw std::invalid_argument("x0 must be non-empty.");
        }
        if (static_cast<int>(x0.size()) != network.input_dim()) {
            throw std::invalid_argument("x0 dimension mismatch with network input.");
        }
        if (x0.size() != eps.size()) {
            throw std::invalid_argument("eps must have same size as x0.");
        }
        for (double e : eps) {
            if (e < 0.0) {
                throw std::invalid_argument("eps must be nonnegative.");
            }
        }
    }

    static AffineBound compose_post_activation_bound(
        const AffineBound& pre,
        const Vec& alpha_l,
        const Vec& beta_l,
        const Vec& alpha_u,
        const Vec& beta_u
    ) {
        const Vec alpha_l_pos = positive_part(alpha_l);
        const Vec alpha_l_neg = negative_part(alpha_l);
        const Vec alpha_u_pos = positive_part(alpha_u);
        const Vec alpha_u_neg = negative_part(alpha_u);

        const Mat lower_A = mat_add(
            rowwise_scale(pre.lower_A, alpha_l_pos),
            rowwise_scale(pre.upper_A, alpha_l_neg)
        );
        const Mat upper_A = mat_add(
            rowwise_scale(pre.upper_A, alpha_u_pos),
            rowwise_scale(pre.lower_A, alpha_u_neg)
        );

        const Vec lower_c = vec_add(vec_add(vec_mul(alpha_l_pos, pre.lower_c), vec_mul(alpha_l_neg, pre.upper_c)), beta_l);
        const Vec upper_c = vec_add(vec_add(vec_mul(alpha_u_pos, pre.upper_c), vec_mul(alpha_u_neg, pre.lower_c)), beta_u);

        return AffineBound{lower_A, lower_c, upper_A, upper_c};
    }

    static std::tuple<Mat, Vec, Mat, Vec> backward_one_layer(
        const Mat& lower_M,
        const Vec& lower_p,
        const Mat& upper_M,
        const Vec& upper_p,
        const Mat& W,
        const Vec& b,
        const Vec& alpha_l,
        const Vec& beta_l,
        const Vec& alpha_u,
        const Vec& beta_u
    ) {
        const Mat lower_M_pos = positive_part(lower_M);
        const Mat lower_M_neg = negative_part(lower_M);
        const Mat upper_M_pos = positive_part(upper_M);
        const Mat upper_M_neg = negative_part(upper_M);

        const Mat lower_s_coeff = mat_add(colwise_scale(lower_M_pos, alpha_l), colwise_scale(lower_M_neg, alpha_u));
        const Mat new_lower_M = matmul(lower_s_coeff, W);

        const Vec alpha_lb = vec_add(vec_mul(alpha_l, b), beta_l);
        const Vec alpha_ub = vec_add(vec_mul(alpha_u, b), beta_u);
        const Vec new_lower_p = vec_add(
            vec_add(matvec(lower_M_pos, alpha_lb), matvec(lower_M_neg, alpha_ub)),
            lower_p
        );

        const Mat upper_s_coeff = mat_add(colwise_scale(upper_M_pos, alpha_u), colwise_scale(upper_M_neg, alpha_l));
        const Mat new_upper_M = matmul(upper_s_coeff, W);

        const Vec new_upper_p = vec_add(
            vec_add(matvec(upper_M_pos, alpha_ub), matvec(upper_M_neg, alpha_lb)),
            upper_p
        );

        return {new_lower_M, new_lower_p, new_upper_M, new_upper_p};
    }

    LayerBound build_one_layer_relaxation(
        const FullyConnectedNetwork& network,
        int layer_index,
        const Vec& x0,
        const Vec& eps,
        const std::vector<LayerBound>& previous_layer_bounds
    ) const {
        Mat lower_M = network.weights()[layer_index];
        Mat upper_M = network.weights()[layer_index];
        Vec lower_p = network.biases()[layer_index];
        Vec upper_p = network.biases()[layer_index];

        for (int prev = layer_index - 1; prev >= 0; --prev) {
            const LayerBound& lb = previous_layer_bounds[prev];
            std::tie(lower_M, lower_p, upper_M, upper_p) = backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                network.weights()[prev], network.biases()[prev],
                lb.alpha_lower, lb.beta_lower, lb.alpha_upper, lb.beta_upper
            );
        }

        AffineBound pre{lower_M, lower_p, upper_M, upper_p};
        Vec pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
        Vec pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

        Vec alpha_l, beta_l, alpha_u, beta_u;
        const std::string& act = network.activations()[layer_index];
        if (act == "linear") {
            linear_relaxation(pre_lower, alpha_l, beta_l, alpha_u, beta_u);
        } else if (act == "relu") {
            ReLURelaxation::relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
        } else if (act == "sigmoid") {
            SigmoidRelaxation().relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
        } else {
            throw std::invalid_argument("Unsupported activation: " + act);
        }

        AffineBound post = compose_post_activation_bound(pre, alpha_l, beta_l, alpha_u, beta_u);
        Vec post_lower = affine_min(post.lower_A, post.lower_c, x0, eps);
        Vec post_upper = affine_max(post.upper_A, post.upper_c, x0, eps);

        return LayerBound{pre, pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u, post, post_lower, post_upper};
    }
};

class LiRPABackwardOnlyIteration {
public:
    std::tuple<AffineBound, Vec, Vec, std::vector<LayerBound>> bound(
        const FullyConnectedNetwork& network,
        const Vec& x0,
        const Vec& eps,
        bool debug = false
    ) const {
        validate_input(network, x0, eps);

        std::vector<LayerBound> layer_bounds = build_layer_relaxations_iterative(network, x0, eps, debug);

        const int output_dim = static_cast<int>(network.weights().back().size());
        Mat lower_M = eye(output_dim);
        Mat upper_M = eye(output_dim);
        Vec lower_p(output_dim, 0.0);
        Vec upper_p(output_dim, 0.0);

        for (int layer_index = static_cast<int>(network.weights().size()) - 1; layer_index >= 0; --layer_index) {
            const LayerBound& layer = layer_bounds[layer_index];
            std::tie(lower_M, lower_p, upper_M, upper_p) = backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                network.weights()[layer_index], network.biases()[layer_index],
                layer.alpha_lower, layer.beta_lower, layer.alpha_upper, layer.beta_upper
            );
        }

        AffineBound final{lower_M, lower_p, upper_M, upper_p};
        Vec final_lower = affine_min(final.lower_A, final.lower_c, x0, eps);
        Vec final_upper = affine_max(final.upper_A, final.upper_c, x0, eps);
        return {final, final_lower, final_upper, layer_bounds};
    }

private:
    static void validate_input(const FullyConnectedNetwork& network, const Vec& x0, const Vec& eps) {
        if (x0.empty()) {
            throw std::invalid_argument("x0 must be non-empty.");
        }
        if (static_cast<int>(x0.size()) != network.input_dim()) {
            throw std::invalid_argument("x0 dimension mismatch with network input.");
        }
        if (x0.size() != eps.size()) {
            throw std::invalid_argument("eps must have same size as x0.");
        }
        for (double e : eps) {
            if (e < 0.0) {
                throw std::invalid_argument("eps must be nonnegative.");
            }
        }
    }

    static void linear_relaxation(const Vec& lower, Vec& alpha_l, Vec& beta_l, Vec& alpha_u, Vec& beta_u) {
        alpha_l = ones_like(lower);
        beta_l = Vec(lower.size(), 0.0);
        alpha_u = alpha_l;
        beta_u = beta_l;
    }

    static std::tuple<Mat, Vec, Mat, Vec> backward_one_layer(
        const Mat& lower_M,
        const Vec& lower_p,
        const Mat& upper_M,
        const Vec& upper_p,
        const Mat& W,
        const Vec& b,
        const Vec& alpha_l,
        const Vec& beta_l,
        const Vec& alpha_u,
        const Vec& beta_u
    ) {
        const Mat lower_M_pos = positive_part(lower_M);
        const Mat lower_M_neg = negative_part(lower_M);
        const Mat upper_M_pos = positive_part(upper_M);
        const Mat upper_M_neg = negative_part(upper_M);

        const Mat lower_s_coeff = mat_add(colwise_scale(lower_M_pos, alpha_l), colwise_scale(lower_M_neg, alpha_u));
        const Mat new_lower_M = matmul(lower_s_coeff, W);

        const Vec alpha_lb = vec_add(vec_mul(alpha_l, b), beta_l);
        const Vec alpha_ub = vec_add(vec_mul(alpha_u, b), beta_u);
        const Vec new_lower_p = vec_add(
            vec_add(matvec(lower_M_pos, alpha_lb), matvec(lower_M_neg, alpha_ub)),
            lower_p
        );

        const Mat upper_s_coeff = mat_add(colwise_scale(upper_M_pos, alpha_u), colwise_scale(upper_M_neg, alpha_l));
        const Mat new_upper_M = matmul(upper_s_coeff, W);

        const Vec new_upper_p = vec_add(
            vec_add(matvec(upper_M_pos, alpha_ub), matvec(upper_M_neg, alpha_lb)),
            upper_p
        );

        return {new_lower_M, new_lower_p, new_upper_M, new_upper_p};
    }

    static AffineBound compose_post_activation_bound(
        const AffineBound& pre,
        const Vec& alpha_l,
        const Vec& beta_l,
        const Vec& alpha_u,
        const Vec& beta_u
    ) {
        const Vec alpha_l_pos = positive_part(alpha_l);
        const Vec alpha_l_neg = negative_part(alpha_l);
        const Vec alpha_u_pos = positive_part(alpha_u);
        const Vec alpha_u_neg = negative_part(alpha_u);

        const Mat lower_A = mat_add(
            rowwise_scale(pre.lower_A, alpha_l_pos),
            rowwise_scale(pre.upper_A, alpha_l_neg)
        );
        const Mat upper_A = mat_add(
            rowwise_scale(pre.upper_A, alpha_u_pos),
            rowwise_scale(pre.lower_A, alpha_u_neg)
        );

        const Vec lower_c = vec_add(vec_add(vec_mul(alpha_l_pos, pre.lower_c), vec_mul(alpha_l_neg, pre.upper_c)), beta_l);
        const Vec upper_c = vec_add(vec_add(vec_mul(alpha_u_pos, pre.upper_c), vec_mul(alpha_u_neg, pre.lower_c)), beta_u);

        return AffineBound{lower_A, lower_c, upper_A, upper_c};
    }

    static void print_vec(const std::string& name, const Vec& v) {
        std::cout << "  " << name << "=[";
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << std::fixed << std::setprecision(6) << v[i];
        }
        std::cout << "]\n";
    }

    static void print_layer_debug_info(int layer_index, const LayerBound& layer_bound) {
        std::cout << "[debug] layer " << (layer_index + 1) << "\n";
        print_vec("pre_lower", layer_bound.pre_lower);
        print_vec("pre_upper", layer_bound.pre_upper);
        print_vec("alpha_lower", layer_bound.alpha_lower);
        print_vec("beta_lower", layer_bound.beta_lower);
        print_vec("alpha_upper", layer_bound.alpha_upper);
        print_vec("beta_upper", layer_bound.beta_upper);
    }

    std::vector<LayerBound> build_layer_relaxations_iterative(
        const FullyConnectedNetwork& network,
        const Vec& x0,
        const Vec& eps,
        bool debug
    ) const {
        std::vector<LayerBound> layer_bounds;
        const int total_layers = static_cast<int>(network.weights().size());

        for (int k = 0; k < total_layers; ++k) {
            Mat lower_C = network.weights()[k];
            Mat upper_C = network.weights()[k];
            Vec lower_d = network.biases()[k];
            Vec upper_d = network.biases()[k];

            for (int r = k - 1; r >= 0; --r) {
                const LayerBound& prev = layer_bounds[r];
                const Mat& W_r = network.weights()[r];
                const Vec& b_r = network.biases()[r];

                const Mat lower_C_pos = positive_part(lower_C);
                const Mat lower_C_neg = negative_part(lower_C);
                lower_C = matmul(
                    mat_add(colwise_scale(lower_C_pos, prev.alpha_lower), colwise_scale(lower_C_neg, prev.alpha_upper)),
                    W_r
                );
                lower_d = vec_add(
                    lower_d,
                    vec_add(
                        matvec(lower_C_pos, vec_add(vec_mul(prev.alpha_lower, b_r), prev.beta_lower)),
                        matvec(lower_C_neg, vec_add(vec_mul(prev.alpha_upper, b_r), prev.beta_upper))
                    )
                );

                const Mat upper_C_pos = positive_part(upper_C);
                const Mat upper_C_neg = negative_part(upper_C);
                upper_C = matmul(
                    mat_add(colwise_scale(upper_C_pos, prev.alpha_upper), colwise_scale(upper_C_neg, prev.alpha_lower)),
                    W_r
                );
                upper_d = vec_add(
                    upper_d,
                    vec_add(
                        matvec(upper_C_pos, vec_add(vec_mul(prev.alpha_upper, b_r), prev.beta_upper)),
                        matvec(upper_C_neg, vec_add(vec_mul(prev.alpha_lower, b_r), prev.beta_lower))
                    )
                );
            }

            AffineBound pre{lower_C, lower_d, upper_C, upper_d};
            Vec pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
            Vec pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

            Vec alpha_l, beta_l, alpha_u, beta_u;
            const std::string& act = network.activations()[k];
            if (act == "linear") {
                linear_relaxation(pre_lower, alpha_l, beta_l, alpha_u, beta_u);
            } else if (act == "relu") {
                ReLURelaxation::relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
            } else if (act == "sigmoid") {
                SigmoidRelaxation().relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
            } else {
                throw std::invalid_argument("Unsupported activation: " + act);
            }

            AffineBound post = compose_post_activation_bound(pre, alpha_l, beta_l, alpha_u, beta_u);
            Vec post_lower = affine_min(post.lower_A, post.lower_c, x0, eps);
            Vec post_upper = affine_max(post.upper_A, post.upper_c, x0, eps);

            layer_bounds.push_back(LayerBound{pre, pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u, post, post_lower, post_upper});

            if (debug) {
                print_layer_debug_info(k, layer_bounds.back());
            }
        }

        return layer_bounds;
    }
};

static FullyConnectedNetwork make_xor_network_from_note() {
    Mat W1 = {
        {2.1247, 2.1267},
        {-2.1237, -2.1235},
    };
    Vec b1 = {-2.1259, 2.1234};

    Mat W2 = {
        {-3.6788, -3.6766},
    };
    Vec b2 = {3.5451};

    return FullyConnectedNetwork(
        {W1, W2},
        {b1, b2},
        {"relu", "sigmoid"}
    );
}

static int xor_expected_label(const Vec& x) {
    if (x.size() < 2) {
        throw std::invalid_argument("xor_expected_label needs at least 2 dimensions.");
    }
    const int a = static_cast<int>(std::llround(x[0]));
    const int b = static_cast<int>(std::llround(x[1]));
    return a ^ b;
}

static bool allclose_vec(const Vec& a, const Vec& b, double atol, double rtol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double tol = atol + rtol * std::abs(b[i]);
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

static void compare_reference_and_iterative(double eps_scalar = 0.02, double atol = 1e-9, double rtol = 1e-9) {
    FullyConnectedNetwork network = make_xor_network_from_note();
    LiRPABackwardOnlyReference ref_verifier;
    LiRPABackwardOnlyIteration itr_verifier;

    std::vector<Vec> points = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    std::cout << "Comparing reference vs iterative backward-only bounds\n";
    std::cout << "Tolerance: atol=" << atol << ", rtol=" << rtol << "\n\n";

    for (const Vec& x0 : points) {
        Vec eps(x0.size(), eps_scalar);

        auto [_, ref_lb, ref_ub, ref_layers] = ref_verifier.bound(network, x0, eps);
        auto [__, itr_lb, itr_ub, itr_layers] = itr_verifier.bound(network, x0, eps, false);

        if (!allclose_vec(ref_lb, itr_lb, atol, rtol)) {
            throw std::runtime_error("Lower bound mismatch between reference and iterative implementation.");
        }
        if (!allclose_vec(ref_ub, itr_ub, atol, rtol)) {
            throw std::runtime_error("Upper bound mismatch between reference and iterative implementation.");
        }
        if (ref_layers.size() != itr_layers.size()) {
            throw std::runtime_error("Layer count mismatch between reference and iterative implementation.");
        }

        for (std::size_t i = 0; i < ref_layers.size(); ++i) {
            if (!allclose_vec(ref_layers[i].pre_lower, itr_layers[i].pre_lower, atol, rtol)) {
                throw std::runtime_error("pre_lower mismatch at layer " + std::to_string(i + 1));
            }
            if (!allclose_vec(ref_layers[i].pre_upper, itr_layers[i].pre_upper, atol, rtol)) {
                throw std::runtime_error("pre_upper mismatch at layer " + std::to_string(i + 1));
            }
            if (!allclose_vec(ref_layers[i].alpha_lower, itr_layers[i].alpha_lower, atol, rtol)) {
                throw std::runtime_error("alpha_lower mismatch at layer " + std::to_string(i + 1));
            }
            if (!allclose_vec(ref_layers[i].beta_lower, itr_layers[i].beta_lower, atol, rtol)) {
                throw std::runtime_error("beta_lower mismatch at layer " + std::to_string(i + 1));
            }
            if (!allclose_vec(ref_layers[i].alpha_upper, itr_layers[i].alpha_upper, atol, rtol)) {
                throw std::runtime_error("alpha_upper mismatch at layer " + std::to_string(i + 1));
            }
            if (!allclose_vec(ref_layers[i].beta_upper, itr_layers[i].beta_upper, atol, rtol)) {
                throw std::runtime_error("beta_upper mismatch at layer " + std::to_string(i + 1));
            }
        }

        std::cout << "x0=[" << x0[0] << ", " << x0[1] << "]: OK\n";
    }

    std::cout << "\nReference and iterative backward-only results match for all tested points.\n";
}

static void run_xor_demo_iterative(double eps_scalar = 0.02, bool debug = false) {
    FullyConnectedNetwork network = make_xor_network_from_note();
    LiRPABackwardOnlyIteration verifier;

    std::vector<Vec> points = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    std::cout << "XOR iterative backward-only CROWN bounds\n";
    std::cout << "Perturbation: L_inf epsilon = " << eps_scalar << "\n\n";

    bool all_certified = true;
    for (const Vec& x0 : points) {
        Vec eps(x0.size(), eps_scalar);
        Vec y = network.forward(x0);

        auto [_, lb, ub, __] = verifier.bound(network, x0, eps, debug);

        const int expected = xor_expected_label(x0);
        bool certified = false;
        std::string condition;
        if (expected == 1) {
            certified = lb[0] > 0.5;
            condition = "lower bound > 0.5";
        } else {
            certified = ub[0] < 0.5;
            condition = "upper bound < 0.5";
        }
        all_certified = all_certified && certified;

        std::cout << std::fixed << std::setprecision(6)
                  << "x0=[" << x0[0] << ", " << x0[1] << "], expected=" << expected
                  << ", network_output=" << y[0] << "\n";
        std::cout << "  iterative backward-only bound=[" << lb[0] << ", " << ub[0] << "]"
                  << ", certified=" << (certified ? "True" : "False")
                  << " (" << condition << ")\n\n";
    }

    if (all_certified) {
        std::cout << "Iterative backward-only mode certifies all four XOR corner classifications for this epsilon.\n";
    } else {
        std::cout << "Iterative backward-only mode does not certify at least one XOR corner classification for this epsilon.\n";
    }
}

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
        run_xor_demo_iterative(eps, debug);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
