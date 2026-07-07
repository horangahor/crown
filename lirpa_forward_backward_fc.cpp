#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <functional>
#include <random>
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

static Mat mat_add(const Mat& A, const Mat& B) {
    if (A.size() != B.size()) {
        throw std::invalid_argument("mat_add row mismatch.");
    }
    if (!A.empty() && A[0].size() != B[0].size()) {
        throw std::invalid_argument("mat_add col mismatch.");
    }
    Mat out = A;
    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            out[i][j] += B[i][j];
        }
    }
    return out;
}

static Mat positive_part(const Mat& A) {
    Mat out = A;
    for (auto& row : out) {
        for (double& x : row) {
            x = std::max(x, 0.0);
        }
    }
    return out;
}

static Mat negative_part(const Mat& A) {
    Mat out = A;
    for (auto& row : out) {
        for (double& x : row) {
            x = std::min(x, 0.0);
        }
    }
    return out;
}

static Vec positive_part(const Vec& v) {
    Vec out = v;
    for (double& x : out) {
        x = std::max(x, 0.0);
    }
    return out;
}

static Vec negative_part(const Vec& v) {
    Vec out = v;
    for (double& x : out) {
        x = std::min(x, 0.0);
    }
    return out;
}

static Mat rowwise_scale(const Mat& A, const Vec& s) {
    if (A.size() != s.size()) {
        throw std::invalid_argument("rowwise_scale mismatch.");
    }
    Mat out = A;
    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[i].size(); ++j) {
            out[i][j] *= s[i];
        }
    }
    return out;
}

static Vec add_bias(const Vec& a, const Vec& b) {
    return vec_add(a, b);
}

static Vec affine_min(const Mat& A, const Vec& c, const Vec& x0, double eps) {
    const Vec x_l = vec_sub(x0, vec_scale(Vec(x0.size(), 1.0), eps));
    const Vec x_u = vec_add(x0, vec_scale(Vec(x0.size(), 1.0), eps));

    if (A.empty()) {
        return c;
    }

    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    if (static_cast<int>(x_l.size()) != n || static_cast<int>(c.size()) != m) {
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

static Vec affine_max(const Mat& A, const Vec& c, const Vec& x0, double eps) {
    const Vec x_l = vec_sub(x0, vec_scale(Vec(x0.size(), 1.0), eps));
    const Vec x_u = vec_add(x0, vec_scale(Vec(x0.size(), 1.0), eps));

    if (A.empty()) {
        return c;
    }

    const int m = static_cast<int>(A.size());
    const int n = static_cast<int>(A[0].size());
    if (static_cast<int>(x_u.size()) != n || static_cast<int>(c.size()) != m) {
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

class ReLURelaxation {
public:
    static double relu(double x) {
        return std::max(0.0, x);
    }

    static void relax(const Vec& lower, const Vec& upper, Vec& alpha_l, Vec& beta_l, Vec& alpha_u, Vec& beta_u) {
        if (lower.size() != upper.size()) {
            throw std::invalid_argument("relu relax size mismatch.");
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
                // Keep zero initialization.
            } else {
                const double denom = u - l;
                alpha_u[i] = u / denom;
                beta_u[i] = -u * l / denom;

                const bool use_identity_lower = std::abs(l) < std::abs(u);
                alpha_l[i] = use_identity_lower ? 1.0 : 0.0;
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
            throw std::invalid_argument("sigmoid relax size mismatch.");
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

            const int samples = 1001;
            double lower_violation = 0.0;
            double upper_violation = 0.0;
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

    double bisect_root(double l, double h, const std::function<double(double)>& fn) const {
        double fl = fn(l);
        double fh = fn(h);

        if (std::abs(fl) < tol_) {
            return l;
        }
        if (std::abs(fh) < tol_) {
            return h;
        }

        if (fl * fh > 0.0) {
            const int grid_n = 257;
            std::vector<double> xs(grid_n);
            std::vector<double> vals(grid_n);
            for (int i = 0; i < grid_n; ++i) {
                xs[i] = l + (h - l) * static_cast<double>(i) / static_cast<double>(grid_n - 1);
                vals[i] = fn(xs[i]);
            }

            int best = 0;
            double best_abs = std::abs(vals[0]);
            for (int i = 1; i < grid_n; ++i) {
                const double cur = std::abs(vals[i]);
                if (cur < best_abs) {
                    best_abs = cur;
                    best = i;
                }
            }

            bool found_bracket = false;
            for (int i = 0; i < grid_n - 1; ++i) {
                if (vals[i] == 0.0 || vals[i] * vals[i + 1] <= 0.0) {
                    l = xs[i];
                    h = xs[i + 1];
                    fl = vals[i];
                    fh = vals[i + 1];
                    found_bracket = true;
                    break;
                }
            }
            if (!found_bracket) {
                return xs[best];
            }
        }

        for (int it = 0; it < max_iter_; ++it) {
            const double m = 0.5 * (l + h);
            const double fm = fn(m);
            if (std::abs(fm) < tol_ || std::abs(h - l) < tol_) {
                return m;
            }
            if (fl * fm <= 0.0) {
                h = m;
                fh = fm;
            } else {
                l = m;
                fl = fm;
            }
        }
        return 0.5 * (l + h);
    }

    double crossing_lower_tangent_point(double l, double u) const {
        const double su = sigma(u);
        const auto fn = [su, u](double d) {
            return (su - sigma(d)) / (u - d) - sigma_prime(d);
        };
        return bisect_root(l, 0.0, fn);
    }

    double crossing_upper_tangent_point(double l, double u) const {
        const double sl = sigma(l);
        const auto fn = [sl, l](double d) {
            return (sigma(d) - sl) / (d - l) - sigma_prime(d);
        };
        return bisect_root(0.0, u, fn);
    }
};

class FullyConnectedNetwork {
public:
    FullyConnectedNetwork(std::vector<Mat> weights, std::vector<Vec> biases, std::vector<std::string> activations)
        : weights_(std::move(weights)), biases_(std::move(biases)), activations_(std::move(activations)) {
        if (!(weights_.size() == biases_.size() && biases_.size() == activations_.size())) {
            throw std::invalid_argument("weights, biases, and activations must have the same length.");
        }

        for (std::size_t layer = 0; layer < weights_.size(); ++layer) {
            check_matrix(weights_[layer], "W");
            if (weights_[layer].empty()) {
                throw std::invalid_argument("Weight matrices must be non-empty.");
            }
            if (weights_[layer].size() != biases_[layer].size()) {
                throw std::invalid_argument("W rows must match bias length.");
            }
            if (layer > 0 && weights_[layer - 1].size() != weights_[layer][0].size()) {
                throw std::invalid_argument("Layer dimension mismatch.");
            }

            for (char& ch : activations_[layer]) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
        }
    }

    int input_dim() const {
        return static_cast<int>(weights_[0][0].size());
    }

    Vec forward(const Vec& x) const {
        Vec f = x;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            Vec s = add_bias(matvec(weights_[i], f), biases_[i]);
            const std::string& act = activations_[i];
            if (act == "relu") {
                for (double& v : s) {
                    v = ReLURelaxation::relu(v);
                }
            } else if (act == "sigmoid") {
                for (double& v : s) {
                    v = SigmoidRelaxation::sigma(v);
                }
            } else if (act == "linear") {
                // Keep s unchanged.
            } else {
                throw std::invalid_argument("Unsupported activation: " + act);
            }
            f = std::move(s);
        }
        return f;
    }

    const std::vector<Mat>& weights() const { return weights_; }
    const std::vector<Vec>& biases() const { return biases_; }
    const std::vector<std::string>& activations() const { return activations_; }

private:
    std::vector<Mat> weights_;
    std::vector<Vec> biases_;
    std::vector<std::string> activations_;
};

class LiRPAForward {
public:
    std::tuple<AffineBound, Vec, Vec, std::vector<LayerBound>> bound(
        const FullyConnectedNetwork& network,
        const Vec& x0,
        double eps
    ) const {
        if (static_cast<int>(x0.size()) != network.input_dim()) {
            throw std::invalid_argument("x0 dimension does not match network input dimension.");
        }

        const int dim = network.input_dim();
        AffineBound current{eye(dim), Vec(dim, 0.0), eye(dim), Vec(dim, 0.0)};
        std::vector<LayerBound> layer_bounds;

        const auto& weights = network.weights();
        const auto& biases = network.biases();
        const auto& acts = network.activations();

        for (std::size_t l = 0; l < weights.size(); ++l) {
            const Mat& W = weights[l];
            const Vec& b = biases[l];
            const std::string& act_name = acts[l];

            Mat W_pos = positive_part(W);
            Mat W_neg = negative_part(W);

            AffineBound pre;
            pre.lower_A = mat_add(matmul(W_pos, current.lower_A), matmul(W_neg, current.upper_A));
            pre.lower_c = vec_add(vec_add(matvec(W_pos, current.lower_c), matvec(W_neg, current.upper_c)), b);
            pre.upper_A = mat_add(matmul(W_pos, current.upper_A), matmul(W_neg, current.lower_A));
            pre.upper_c = vec_add(vec_add(matvec(W_pos, current.upper_c), matvec(W_neg, current.lower_c)), b);

            Vec pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps);
            Vec pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps);

            Vec alpha_l, beta_l, alpha_u, beta_u;
            if (act_name == "linear") {
                const int n = static_cast<int>(pre_lower.size());
                alpha_l.assign(n, 1.0);
                alpha_u.assign(n, 1.0);
                beta_l.assign(n, 0.0);
                beta_u.assign(n, 0.0);
            } else if (act_name == "relu") {
                ReLURelaxation::relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
            } else if (act_name == "sigmoid") {
                SigmoidRelaxation().relax(pre_lower, pre_upper, alpha_l, beta_l, alpha_u, beta_u);
            } else {
                throw std::invalid_argument("No relaxation for activation: " + act_name);
            }

            AffineBound post;
            post.lower_A = rowwise_scale(pre.lower_A, alpha_l);
            post.lower_c = vec_add(vec_mul(alpha_l, pre.lower_c), beta_l);
            post.upper_A = rowwise_scale(pre.upper_A, alpha_u);
            post.upper_c = vec_add(vec_mul(alpha_u, pre.upper_c), beta_u);

            Vec post_lower = affine_min(post.lower_A, post.lower_c, x0, eps);
            Vec post_upper = affine_max(post.upper_A, post.upper_c, x0, eps);

            layer_bounds.push_back(LayerBound{
                pre, pre_lower, pre_upper,
                alpha_l, beta_l, alpha_u, beta_u,
                post, post_lower, post_upper
            });
            current = post;
        }

        Vec final_lower = affine_min(current.lower_A, current.lower_c, x0, eps);
        Vec final_upper = affine_max(current.upper_A, current.upper_c, x0, eps);
        return {current, final_lower, final_upper, layer_bounds};
    }
};

class LiRPABackward {
public:
    explicit LiRPABackward(const LiRPAForward* forward_verifier = nullptr)
        : forward_verifier_(forward_verifier ? *forward_verifier : LiRPAForward()) {}

    std::tuple<AffineBound, Vec, Vec, std::vector<LayerBound>> bound(
        const FullyConnectedNetwork& network,
        const Vec& x0,
        double eps,
        const Mat* output_lower_M = nullptr,
        const Vec* output_lower_p = nullptr,
        const Mat* output_upper_M = nullptr,
        const Vec* output_upper_p = nullptr
    ) const {
        if (static_cast<int>(x0.size()) != network.input_dim()) {
            throw std::invalid_argument("x0 dimension does not match network input dimension.");
        }

        auto [unused_affine, unused_lb, unused_ub, layer_bounds] = forward_verifier_.bound(network, x0, eps);
        (void)unused_affine;
        (void)unused_lb;
        (void)unused_ub;

        const int output_dim = static_cast<int>(network.weights().back().size());

        Mat lower_M = output_lower_M ? *output_lower_M : eye(output_dim);
        Mat upper_M = output_upper_M ? *output_upper_M : eye(output_dim);

        const int spec_dim = static_cast<int>(lower_M.size());
        Vec lower_p = output_lower_p ? *output_lower_p : Vec(spec_dim, 0.0);
        Vec upper_p = output_upper_p ? *output_upper_p : Vec(static_cast<int>(upper_M.size()), 0.0);

        if (lower_M.empty() || upper_M.empty()) {
            throw std::invalid_argument("Output specification matrices must be non-empty.");
        }
        if (static_cast<int>(lower_M[0].size()) != output_dim || static_cast<int>(upper_M[0].size()) != output_dim) {
            throw std::invalid_argument("Output specification matrices must have one column per network output.");
        }
        if (lower_p.size() != lower_M.size() || upper_p.size() != upper_M.size()) {
            throw std::invalid_argument("Output specification vectors must match matrix rows.");
        }
        if (lower_M.size() != upper_M.size()) {
            throw std::invalid_argument("Lower and upper output specifications must have same row count.");
        }

        const auto& weights = network.weights();
        const auto& biases = network.biases();

        for (int i = static_cast<int>(weights.size()) - 1; i >= 0; --i) {
            std::tie(lower_M, lower_p, upper_M, upper_p) = backward_one_layer(
                lower_M, lower_p, upper_M, upper_p,
                weights[i], biases[i],
                layer_bounds[i].alpha_lower,
                layer_bounds[i].beta_lower,
                layer_bounds[i].alpha_upper,
                layer_bounds[i].beta_upper
            );
        }

        AffineBound final{lower_M, lower_p, upper_M, upper_p};
        Vec final_lower = affine_min(final.lower_A, final.lower_c, x0, eps);
        Vec final_upper = affine_max(final.upper_A, final.upper_c, x0, eps);
        return {final, final_lower, final_upper, layer_bounds};
    }

private:
    LiRPAForward forward_verifier_;

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
        Mat lower_M_pos = positive_part(lower_M);
        Mat lower_M_neg = negative_part(lower_M);
        Mat upper_M_pos = positive_part(upper_M);
        Mat upper_M_neg = negative_part(upper_M);

        Mat lower_s_coeff = lower_M;
        Mat upper_s_coeff = upper_M;
        for (std::size_t i = 0; i < lower_s_coeff.size(); ++i) {
            for (std::size_t j = 0; j < lower_s_coeff[i].size(); ++j) {
                lower_s_coeff[i][j] = lower_M_pos[i][j] * alpha_l[j] + lower_M_neg[i][j] * alpha_u[j];
                upper_s_coeff[i][j] = upper_M_pos[i][j] * alpha_u[j] + upper_M_neg[i][j] * alpha_l[j];
            }
        }

        Mat new_lower_M = matmul(lower_s_coeff, W);
        Mat new_upper_M = matmul(upper_s_coeff, W);

        Vec term_l_pos = vec_add(vec_mul(alpha_l, b), beta_l);
        Vec term_l_neg = vec_add(vec_mul(alpha_u, b), beta_u);
        Vec new_lower_p = vec_add(vec_add(matvec(lower_M_pos, term_l_pos), matvec(lower_M_neg, term_l_neg)), lower_p);

        Vec term_u_pos = vec_add(vec_mul(alpha_u, b), beta_u);
        Vec term_u_neg = vec_add(vec_mul(alpha_l, b), beta_l);
        Vec new_upper_p = vec_add(vec_add(matvec(upper_M_pos, term_u_pos), matvec(upper_M_neg, term_u_neg)), upper_p);

        return {new_lower_M, new_lower_p, new_upper_M, new_upper_p};
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

    return FullyConnectedNetwork({W1, W2}, {b1, b2}, {"relu", "sigmoid"});
}

static int xor_expected_label(const Vec& x) {
    const int a = static_cast<int>(std::llround(x[0]));
    const int b = static_cast<int>(std::llround(x[1]));
    return a ^ b;
}

static void self_test_relaxations() {
    std::mt19937_64 rng(0);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    SigmoidRelaxation sig_relax;

    for (int t = 0; t < 200; ++t) {
        double a = dist(rng);
        double b = dist(rng);
        if (a > b) {
            std::swap(a, b);
        }
        if (std::abs(a - b) < 1e-8) {
            b = a + 1e-6;
        }

        Vec l = {a};
        Vec u = {b};

        {
            Vec al, bl, au, bu;
            ReLURelaxation::relax(l, u, al, bl, au, bu);
            for (int k = 0; k < 201; ++k) {
                double x = a + (b - a) * static_cast<double>(k) / 200.0;
                double y = ReLURelaxation::relu(x);
                double lhs = al[0] * x + bl[0];
                double rhs = au[0] * x + bu[0];
                if (lhs > y + 1e-8 || y > rhs + 1e-8) {
                    throw std::runtime_error("ReLU relaxation self-test failed.");
                }
            }
        }

        {
            Vec al, bl, au, bu;
            sig_relax.relax(l, u, al, bl, au, bu);
            for (int k = 0; k < 201; ++k) {
                double x = a + (b - a) * static_cast<double>(k) / 200.0;
                double y = SigmoidRelaxation::sigma(x);
                double lhs = al[0] * x + bl[0];
                double rhs = au[0] * x + bu[0];
                if (lhs > y + 1e-8 || y > rhs + 1e-8) {
                    throw std::runtime_error("Sigmoid relaxation self-test failed.");
                }
            }
        }
    }
}

static void run_xor_demo(double eps = 0.02) {
    FullyConnectedNetwork network = make_xor_network_from_note();
    LiRPAForward forward_verifier;
    LiRPABackward backward_verifier(&forward_verifier);

    std::vector<Vec> points = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    std::cout << "XOR network point predictions and LiRPA-certified output bounds\n";
    std::cout << "Perturbation: L_inf epsilon = " << eps << "\n\n";

    bool all_certified_forward = true;
    bool all_certified_backward = true;

    std::cout << std::fixed << std::setprecision(6);

    for (const Vec& x0 : points) {
        Vec y = network.forward(x0);

        auto [fwd_affine, fwd_lb, fwd_ub, fwd_layers] = forward_verifier.bound(network, x0, eps);
        (void)fwd_affine;
        (void)fwd_layers;

        auto [bwd_affine, bwd_lb, bwd_ub, bwd_layers] = backward_verifier.bound(network, x0, eps);
        (void)bwd_affine;
        (void)bwd_layers;

        int expected = xor_expected_label(x0);
        bool fwd_certified;
        bool bwd_certified;
        std::string condition;

        if (expected == 1) {
            fwd_certified = (fwd_lb[0] > 0.5);
            bwd_certified = (bwd_lb[0] > 0.5);
            condition = "lower bound > 0.5";
        } else {
            fwd_certified = (fwd_ub[0] < 0.5);
            bwd_certified = (bwd_ub[0] < 0.5);
            condition = "upper bound < 0.5";
        }

        all_certified_forward = all_certified_forward && fwd_certified;
        all_certified_backward = all_certified_backward && bwd_certified;

        std::cout << "x0=[" << x0[0] << ", " << x0[1] << "], expected=" << expected
                  << ", network_output=" << y[0] << "\n";
        std::cout << "  forward  bound=[" << fwd_lb[0] << ", " << fwd_ub[0] << "], certified="
                  << (fwd_certified ? "True" : "False") << " (" << condition << ")\n";
        std::cout << "  backward bound=[" << bwd_lb[0] << ", " << bwd_ub[0] << "], certified="
                  << (bwd_certified ? "True" : "False") << " (" << condition << ")\n";
    }

    std::cout << "\n";
    if (all_certified_forward) {
        std::cout << "Forward mode certifies all four XOR corner classifications for this epsilon.\n";
    } else {
        std::cout << "Forward mode does not certify at least one XOR corner classification for this epsilon.\n";
    }

    if (all_certified_backward) {
        std::cout << "Backward mode certifies all four XOR corner classifications for this epsilon.\n";
    } else {
        std::cout << "Backward mode does not certify at least one XOR corner classification for this epsilon.\n";
    }
}

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
