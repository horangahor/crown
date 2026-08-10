"""
A small, educational LiRPA forward/backward-mode implementation for fully-connected
networks with ReLU hidden activations and a Sigmoid output activation.

The implementation follows the notation used in the uploaded LiRPA note:
    s^(l) = W^(l) f^(l-1) + b^(l)
    f^(l) = sigma^(l)(s^(l))

Forward mode propagates symbolic affine lower/upper bounds of the form
    A_lower x + c_lower <= f <= A_upper x + c_upper

Backward mode first obtains activation slopes/intercepts from forward mode
and then propagates a symbolic output specification backward to the input.

Currently supported activation relaxations:
    - ReLU
    - Sigmoid

The activation relaxation interface is intentionally modular so that other
monotone activations can be added later.

Run:
    python lirpa_forward_fc.py

Dependencies:
    numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence, Tuple
import math
import numpy as np


Array = np.ndarray


@dataclass
class AffineBound:
    """Symbolic affine lower and upper bounds."""
    lower_A: Array
    lower_c: Array
    upper_A: Array
    upper_c: Array


@dataclass
class LayerBound:
    """All useful bounds for one network layer."""
    pre_affine: AffineBound
    pre_lower: Array
    pre_upper: Array
    alpha_lower: Array
    beta_lower: Array
    alpha_upper: Array
    beta_upper: Array
    post_affine: AffineBound
    post_lower: Array
    post_upper: Array


class ActivationRelaxation(Protocol):
    """Interface for activation-specific linear relaxations."""

    name: str

    def relax(self, lower: Array, upper: Array) -> Tuple[Array, Array, Array, Array]:
        """
        Return alpha_lower, beta_lower, alpha_upper, beta_upper such that
            alpha_lower * s + beta_lower <= sigma(s)
            sigma(s) <= alpha_upper * s + beta_upper
        for every s in [lower, upper].
        """
        ...


class ReLURelaxation:
    """Linear relaxation for ReLU on an interval [l, u]."""

    name = "relu"

    @staticmethod
    def relu(x: Array | float) -> Array | float:
        x_arr = np.asarray(x, dtype=float)
        out = np.maximum(x_arr, 0.0)
        if np.isscalar(x):
            return float(out)
        return out

    def relax(self, lower: Array, upper: Array) -> Tuple[Array, Array, Array, Array]:
        l = np.asarray(lower, dtype=float)
        u = np.asarray(upper, dtype=float)
        if np.any(l > u):
            raise ValueError("Invalid interval: lower must be <= upper.")

        alpha_l = np.zeros_like(l)
        beta_l = np.zeros_like(l)
        alpha_u = np.zeros_like(l)
        beta_u = np.zeros_like(l)

        positive = l >= 0.0
        negative = u <= 0.0
        crossing = ~(positive | negative)

        # Fully active: ReLU(s) = s.
        alpha_l[positive] = 1.0
        alpha_u[positive] = 1.0

        # Fully inactive: ReLU(s) = 0. Already initialized to zero.

        # Crossing interval: l < 0 < u.
        idx = crossing
        denom = u[idx] - l[idx]
        alpha_u[idx] = u[idx] / denom
        beta_u[idx] = -u[idx] * l[idx] / denom

        # Common CROWN/DeepPoly lower-bound choice:
        # use slope 1 if the positive side is wider, otherwise slope 0.
        use_identity_lower = np.abs(l[idx]) < np.abs(u[idx])
        alpha_l[idx] = use_identity_lower.astype(float)
        beta_l[idx] = 0.0

        return alpha_l, beta_l, alpha_u, beta_u


class SigmoidRelaxation:
    """
    Linear relaxation for Sigmoid on an interval [l, u].

    Sigmoid is convex on (-inf, 0] and concave on [0, inf).
    For same-sign intervals, this implementation uses the standard
    secant/tangent construction. For crossing intervals, it finds tangent
    points by bisection, following the equations commonly used in CROWN.
    """

    name = "sigmoid"

    def __init__(self, max_iter: int = 80, tol: float = 1e-12):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def sigma(x: Array | float) -> Array | float:
        # Numerically stable sigmoid.
        x_arr = np.asarray(x, dtype=float)
        out = np.empty_like(x_arr)
        pos = x_arr >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
        exp_x = np.exp(x_arr[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        if np.isscalar(x):
            return float(out)
        return out

    @classmethod
    def sigma_prime(cls, x: Array | float) -> Array | float:
        s = cls.sigma(x)
        return s * (1.0 - s)

    def _bisect_root(self, fn, lo: float, hi: float) -> float:
        flo = fn(lo)
        fhi = fn(hi)

        if abs(flo) < self.tol:
            return lo
        if abs(fhi) < self.tol:
            return hi

        # If a clean bracket is unavailable due to numerical degeneracy,
        # fall back to a small grid and choose the best sign-change bracket.
        if flo * fhi > 0:
            xs = np.linspace(lo, hi, 257)
            vals = np.array([fn(float(x)) for x in xs])
            best = int(np.argmin(np.abs(vals)))
            for i in range(len(xs) - 1):
                if vals[i] == 0 or vals[i] * vals[i + 1] <= 0:
                    lo, hi = float(xs[i]), float(xs[i + 1])
                    flo, fhi = float(vals[i]), float(vals[i + 1])
                    break
            else:
                return float(xs[best])

        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            fmid = fn(mid)
            if abs(fmid) < self.tol or abs(hi - lo) < self.tol:
                return mid
            if flo * fmid <= 0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        return 0.5 * (lo + hi)

    def _crossing_lower_tangent_point(self, l: float, u: float) -> float:
        # Find d_u in [l, 0] satisfying
        # (sigma(u) - sigma(d_u)) / (u - d_u) = sigma'(d_u).
        su = self.sigma(u)

        def fn(d: float) -> float:
            return (su - self.sigma(d)) / (u - d) - self.sigma_prime(d)

        return self._bisect_root(fn, l, 0.0)

    def _crossing_upper_tangent_point(self, l: float, u: float) -> float:
        # Find d_l in [0, u] satisfying
        # (sigma(d_l) - sigma(l)) / (d_l - l) = sigma'(d_l).
        sl = self.sigma(l)

        def fn(d: float) -> float:
            return (self.sigma(d) - sl) / (d - l) - self.sigma_prime(d)

        return self._bisect_root(fn, 0.0, u)

    def relax(self, lower: Array, upper: Array) -> Tuple[Array, Array, Array, Array]:
        l = np.asarray(lower, dtype=float)
        u = np.asarray(upper, dtype=float)
        if np.any(l > u):
            raise ValueError("Invalid interval: lower must be <= upper.")

        alpha_l = np.zeros_like(l)
        beta_l = np.zeros_like(l)
        alpha_u = np.zeros_like(l)
        beta_u = np.zeros_like(l)

        flat_l = l.reshape(-1)
        flat_u = u.reshape(-1)
        al = alpha_l.reshape(-1)
        bl = beta_l.reshape(-1)
        au = alpha_u.reshape(-1)
        bu = beta_u.reshape(-1)

        for i, (li, ui) in enumerate(zip(flat_l, flat_u)):
            if abs(ui - li) < 1e-14:
                slope = float(self.sigma_prime(li))
                intercept = float(self.sigma(li) - slope * li)
                al[i] = au[i] = slope
                bl[i] = bu[i] = intercept
                continue

            if li >= 0.0:
                # Concave region: secant is lower, tangent is upper.
                slope_sec = float((self.sigma(ui) - self.sigma(li)) / (ui - li))
                al[i] = slope_sec
                bl[i] = float(self.sigma(ui) - slope_sec * ui)

                x0 = 0.5 * (li + ui)
                slope_tan = float(self.sigma_prime(x0))
                au[i] = slope_tan
                bu[i] = float(self.sigma(x0) - slope_tan * x0)

            elif ui <= 0.0:
                # Convex region: tangent is lower, secant is upper.
                x0 = 0.5 * (li + ui)
                slope_tan = float(self.sigma_prime(x0))
                al[i] = slope_tan
                bl[i] = float(self.sigma(x0) - slope_tan * x0)

                slope_sec = float((self.sigma(ui) - self.sigma(li)) / (ui - li))
                au[i] = slope_sec
                bu[i] = float(self.sigma(ui) - slope_sec * ui)

            else:
                # Crossing interval: use tangent points found by bisection.
                du = self._crossing_lower_tangent_point(li, ui)
                dl = self._crossing_upper_tangent_point(li, ui)

                slope_lower = float(self.sigma_prime(du))
                al[i] = slope_lower
                bl[i] = float(self.sigma(du) - slope_lower * du)

                slope_upper = float(self.sigma_prime(dl))
                au[i] = slope_upper
                bu[i] = float(self.sigma(dl) - slope_upper * dl)

            # Numerical safety repair.  The closed-form cases above are the
            # intended LiRPA/CROWN relaxations, but bisection can suffer from
            # boundary degeneracy on very asymmetric crossing intervals.  We
            # conservatively shift the intercepts if dense samples detect a
            # tiny violation.  This keeps the implementation robust while
            # preserving the activation-specific modular structure.
            xs = np.linspace(li, ui, 1001)
            ys = self.sigma(xs)

            lower_line = al[i] * xs + bl[i]
            lower_violation = float(np.max(lower_line - ys))
            if lower_violation > 1e-10:
                bl[i] -= lower_violation + 1e-10

            upper_line = au[i] * xs + bu[i]
            upper_violation = float(np.max(ys - upper_line))
            if upper_violation > 1e-10:
                bu[i] += upper_violation + 1e-10

        return alpha_l, beta_l, alpha_u, beta_u


def positive_part(x: Array) -> Array:
    return np.maximum(x, 0.0)


def negative_part(x: Array) -> Array:
    return np.minimum(x, 0.0)


def affine_min(A: Array, c: Array, x0: Array, eps: float | Array) -> Array:
    """Minimize A x + c over x in [x0 - eps, x0 + eps]."""
    x0 = np.asarray(x0, dtype=float)
    eps = np.asarray(eps, dtype=float)
    x_l = x0 - eps
    x_u = x0 + eps
    return positive_part(A) @ x_l + negative_part(A) @ x_u + c


def affine_max(A: Array, c: Array, x0: Array, eps: float | Array) -> Array:
    """Maximize A x + c over x in [x0 - eps, x0 + eps]."""
    x0 = np.asarray(x0, dtype=float)
    eps = np.asarray(eps, dtype=float)
    x_l = x0 - eps
    x_u = x0 + eps
    return positive_part(A) @ x_u + negative_part(A) @ x_l + c


class FullyConnectedNetwork:
    """A simple fully-connected network with per-layer activation names."""

    def __init__(self, weights: Sequence[Array], biases: Sequence[Array], activations: Sequence[str]):
        if not (len(weights) == len(biases) == len(activations)):
            raise ValueError("weights, biases, and activations must have the same length.")

        self.weights = [np.asarray(W, dtype=float) for W in weights]
        self.biases = [np.asarray(b, dtype=float) for b in biases]
        self.activations = [a.lower() for a in activations]

        for layer, (W, b) in enumerate(zip(self.weights, self.biases), start=1):
            if W.ndim != 2:
                raise ValueError(f"W[{layer}] must be a matrix.")
            if b.ndim != 1:
                raise ValueError(f"b[{layer}] must be a vector.")
            if W.shape[0] != b.shape[0]:
                raise ValueError(f"W[{layer}] rows must match b[{layer}] length.")
            if layer > 1 and self.weights[layer - 2].shape[0] != W.shape[1]:
                raise ValueError(f"Layer dimension mismatch before layer {layer}.")

    @property
    def input_dim(self) -> int:
        return self.weights[0].shape[1]

    def forward(self, x: Array) -> Array:
        f = np.asarray(x, dtype=float)
        for W, b, act in zip(self.weights, self.biases, self.activations):
            s = W @ f + b
            if act == "relu":
                f = ReLURelaxation.relu(s)
            elif act == "sigmoid":
                f = SigmoidRelaxation.sigma(s)
            elif act == "linear":
                f = s
            else:
                raise ValueError(f"Unsupported activation: {act}")
        return f


class LiRPAForward:
    """Forward-mode LiRPA bound propagation for a FullyConnectedNetwork."""

    def __init__(self, activation_relaxations: Dict[str, ActivationRelaxation] | None = None):
        self.activation_relaxations: Dict[str, ActivationRelaxation] = {
            "relu": ReLURelaxation(),
            "sigmoid": SigmoidRelaxation(),
        }
        if activation_relaxations:
            self.activation_relaxations.update({k.lower(): v for k, v in activation_relaxations.items()})

    def _linear_relaxation(self, lower: Array, upper: Array) -> Tuple[Array, Array, Array, Array]:
        alpha = np.ones_like(lower)
        beta = np.zeros_like(lower)
        return alpha, beta, alpha, beta

    def bound(self, network: FullyConnectedNetwork, x0: Array, eps: float | Array) -> Tuple[AffineBound, Array, Array, List[LayerBound]]:
        """
        Compute final output bounds for all x in [x0 - eps, x0 + eps].

        Returns:
            final_affine_bound, numerical_lower, numerical_upper, per_layer_bounds
        """
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim != 1:
            raise ValueError("x0 must be a vector.")
        if x0.shape[0] != network.input_dim:
            raise ValueError(f"x0 dimension {x0.shape[0]} does not match network input dimension {network.input_dim}.")

        # f^(0) = x, so the initial symbolic lower/upper bounds are exact.
        dim = network.input_dim
        current = AffineBound(
            lower_A=np.eye(dim),
            lower_c=np.zeros(dim),
            upper_A=np.eye(dim),
            upper_c=np.zeros(dim),
        )

        layer_bounds: List[LayerBound] = []

        for W, b, act_name in zip(network.weights, network.biases, network.activations):
            W_pos = positive_part(W)
            W_neg = negative_part(W)

            # Bound pre-activation s = W f + b using conditional multiplication.
            pre = AffineBound(
                lower_A=W_pos @ current.lower_A + W_neg @ current.upper_A,
                lower_c=W_pos @ current.lower_c + W_neg @ current.upper_c + b,
                upper_A=W_pos @ current.upper_A + W_neg @ current.lower_A,
                upper_c=W_pos @ current.upper_c + W_neg @ current.lower_c + b,
            )

            pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps)
            pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps)

            if act_name == "linear":
                alpha_l, beta_l, alpha_u, beta_u = self._linear_relaxation(pre_lower, pre_upper)
            else:
                relaxation = self.activation_relaxations.get(act_name)
                if relaxation is None:
                    raise ValueError(f"No relaxation registered for activation: {act_name}")
                alpha_l, beta_l, alpha_u, beta_u = relaxation.relax(pre_lower, pre_upper)

            # For ReLU/Sigmoid, slopes are nonnegative, so multiplying the
            # symbolic lower/upper pre-activation bounds is safe.
            post = AffineBound(
                lower_A=alpha_l[:, None] * pre.lower_A,
                lower_c=alpha_l * pre.lower_c + beta_l,
                upper_A=alpha_u[:, None] * pre.upper_A,
                upper_c=alpha_u * pre.upper_c + beta_u,
            )

            post_lower = affine_min(post.lower_A, post.lower_c, x0, eps)
            post_upper = affine_max(post.upper_A, post.upper_c, x0, eps)

            layer_bounds.append(
                LayerBound(
                    pre_affine=pre,
                    pre_lower=pre_lower,
                    pre_upper=pre_upper,
                    alpha_lower=alpha_l,
                    beta_lower=beta_l,
                    alpha_upper=alpha_u,
                    beta_upper=beta_u,
                    post_affine=post,
                    post_lower=post_lower,
                    post_upper=post_upper,
                )
            )
            current = post

        final_lower = affine_min(current.lower_A, current.lower_c, x0, eps)
        final_upper = affine_max(current.upper_A, current.upper_c, x0, eps)
        return current, final_lower, final_upper, layer_bounds



class LiRPABackward:
    """
    Backward-mode LiRPA bound propagation for a FullyConnectedNetwork.

    This class intentionally reuses the activation relaxation information
    computed during a forward pass.  In particular, for each layer l, the
    forward pass gives vectors

        alpha_lower, beta_lower, alpha_upper, beta_upper

    satisfying

        alpha_lower * s^(l) + beta_lower <= f^(l)
        f^(l) <= alpha_upper * s^(l) + beta_upper.

    Starting from the identity bound on the final output f^(L), backward mode
    repeatedly substitutes the activation relaxation and then

        s^(l) = W^(l) f^(l-1) + b^(l)

    to obtain affine bounds directly with respect to the original input.
    """

    def __init__(self, forward_verifier: LiRPAForward | None = None):
        self.forward_verifier = forward_verifier or LiRPAForward()

    @staticmethod
    def _backward_one_layer(
        lower_M: Array,
        lower_p: Array,
        upper_M: Array,
        upper_p: Array,
        W: Array,
        b: Array,
        alpha_l: Array,
        beta_l: Array,
        alpha_u: Array,
        beta_u: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """
        Convert bounds over f^(l) into bounds over f^(l-1).

        Given
            lower_M f^(l) + lower_p <= y <= upper_M f^(l) + upper_p,
        and linear activation bounds over f^(l), this implements conditional
        multiplication with the positive/negative parts of lower_M and upper_M.
        """
        lower_M_pos = positive_part(lower_M)
        lower_M_neg = negative_part(lower_M)
        upper_M_pos = positive_part(upper_M)
        upper_M_neg = negative_part(upper_M)

        # Lower side: positive coefficients use the activation lower bound;
        # negative coefficients use the activation upper bound.
        lower_s_coeff = lower_M_pos * alpha_l[None, :] + lower_M_neg * alpha_u[None, :]
        new_lower_M = lower_s_coeff @ W
        new_lower_p = (
            lower_M_pos @ (alpha_l * b + beta_l)
            + lower_M_neg @ (alpha_u * b + beta_u)
            + lower_p
        )

        # Upper side: positive coefficients use the activation upper bound;
        # negative coefficients use the activation lower bound.
        upper_s_coeff = upper_M_pos * alpha_u[None, :] + upper_M_neg * alpha_l[None, :]
        new_upper_M = upper_s_coeff @ W
        new_upper_p = (
            upper_M_pos @ (alpha_u * b + beta_u)
            + upper_M_neg @ (alpha_l * b + beta_l)
            + upper_p
        )

        return new_lower_M, new_lower_p, new_upper_M, new_upper_p

    def bound(
        self,
        network: FullyConnectedNetwork,
        x0: Array,
        eps: float | Array,
        output_lower_M: Array | None = None,
        output_lower_p: Array | None = None,
        output_upper_M: Array | None = None,
        output_upper_p: Array | None = None,
    ) -> Tuple[AffineBound, Array, Array, List[LayerBound]]:
        """
        Compute backward-mode LiRPA bounds for all x in [x0 - eps, x0 + eps].

        By default this bounds the network output itself, using
            I f^(L) + 0 <= f^(L) <= I f^(L) + 0.

        A custom output specification can also be supplied.  For example, for a
        two-output classifier one may set a row vector such as e_y - e_t to
        bound a logit margin.

        Returns:
            final_affine_bound, numerical_lower, numerical_upper, per_layer_bounds
        """
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim != 1:
            raise ValueError("x0 must be a vector.")
        if x0.shape[0] != network.input_dim:
            raise ValueError(f"x0 dimension {x0.shape[0]} does not match network input dimension {network.input_dim}.")

        # Forward mode is used only to obtain valid pre-activation intervals and
        # the corresponding activation relaxation slopes/intercepts.
        _, _, _, layer_bounds = self.forward_verifier.bound(network, x0, eps)

        output_dim = network.weights[-1].shape[0]
        if output_lower_M is None:
            lower_M = np.eye(output_dim)
        else:
            lower_M = np.asarray(output_lower_M, dtype=float)
        if output_upper_M is None:
            upper_M = np.eye(output_dim)
        else:
            upper_M = np.asarray(output_upper_M, dtype=float)

        spec_dim = lower_M.shape[0]
        if output_lower_p is None:
            lower_p = np.zeros(spec_dim)
        else:
            lower_p = np.asarray(output_lower_p, dtype=float)
        if output_upper_p is None:
            upper_p = np.zeros(upper_M.shape[0])
        else:
            upper_p = np.asarray(output_upper_p, dtype=float)

        if lower_M.shape[1] != output_dim or upper_M.shape[1] != output_dim:
            raise ValueError("Output specification matrices must have one column per network output.")
        if lower_p.shape[0] != lower_M.shape[0] or upper_p.shape[0] != upper_M.shape[0]:
            raise ValueError("Output specification vectors must match their matrices.")
        if lower_M.shape[0] != upper_M.shape[0]:
            raise ValueError("Lower and upper output specifications must have the same number of rows.")

        for W, b, lb in reversed(list(zip(network.weights, network.biases, layer_bounds))):
            lower_M, lower_p, upper_M, upper_p = self._backward_one_layer(
                lower_M=lower_M,
                lower_p=lower_p,
                upper_M=upper_M,
                upper_p=upper_p,
                W=W,
                b=b,
                alpha_l=lb.alpha_lower,
                beta_l=lb.beta_lower,
                alpha_u=lb.alpha_upper,
                beta_u=lb.beta_upper,
            )

        final = AffineBound(
            lower_A=lower_M,
            lower_c=lower_p,
            upper_A=upper_M,
            upper_c=upper_p,
        )
        final_lower = affine_min(final.lower_A, final.lower_c, x0, eps)
        final_upper = affine_max(final.upper_A, final.upper_c, x0, eps)
        return final, final_lower, final_upper, layer_bounds

class LiRPABackwardOnly:
    """
    Backward-only CROWN/LiRPA for a FullyConnectedNetwork.

    Unlike :class:`LiRPABackward`, this class never invokes forward LiRPA to
    obtain activation intervals.  To construct the relaxation of layer k, it
    bounds the pre-activation

        s^(k) = W^(k) f^(k-1) + b^(k)

    by back-substituting through the already constructed relaxations of layers
    1, ..., k-1 all the way to the original input.  The resulting affine lower
    and upper bounds are concretized over [x0-eps, x0+eps], and only then is the
    relaxation for activation k constructed.

    Consequently, relaxation construction proceeds from the first layer to the
    last, but every intermediate pre-activation bound is itself obtained by a
    backward pass to the input.  After all relaxations have been constructed, a
    final backward pass evaluates the requested output specification.
    """

    def __init__(
        self,
        activation_relaxations: Dict[str, ActivationRelaxation] | None = None,
    ):
        self.activation_relaxations: Dict[str, ActivationRelaxation] = {
            "relu": ReLURelaxation(),
            "sigmoid": SigmoidRelaxation(),
        }
        if activation_relaxations:
            self.activation_relaxations.update(
                {name.lower(): relaxation for name, relaxation in activation_relaxations.items()}
            )

    @staticmethod
    def _linear_relaxation(
        lower: Array,
        upper: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """Return the exact relaxation for the identity activation."""
        del upper  # Kept in the interface for consistency with nonlinear relaxations.
        alpha = np.ones_like(lower)
        beta = np.zeros_like(lower)
        return alpha, beta, alpha.copy(), beta.copy()

    @staticmethod
    def _backward_one_layer(
        lower_M: Array,
        lower_p: Array,
        upper_M: Array,
        upper_p: Array,
        W: Array,
        b: Array,
        alpha_l: Array,
        beta_l: Array,
        alpha_u: Array,
        beta_u: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        """Back-substitute affine bounds through one activation and linear layer."""
        lower_M_pos = positive_part(lower_M)
        lower_M_neg = negative_part(lower_M)
        upper_M_pos = positive_part(upper_M)
        upper_M_neg = negative_part(upper_M)

        # Lower expression: positive coefficients select the activation lower
        # relaxation, while negative coefficients select the upper relaxation.
        lower_s_coeff = (
            lower_M_pos * alpha_l[None, :]
            + lower_M_neg * alpha_u[None, :]
        )
        new_lower_M = lower_s_coeff @ W
        new_lower_p = (
            lower_M_pos @ (alpha_l * b + beta_l)
            + lower_M_neg @ (alpha_u * b + beta_u)
            + lower_p
        )

        # Upper expression: positive coefficients select the activation upper
        # relaxation, while negative coefficients select the lower relaxation.
        upper_s_coeff = (
            upper_M_pos * alpha_u[None, :]
            + upper_M_neg * alpha_l[None, :]
        )
        new_upper_M = upper_s_coeff @ W
        new_upper_p = (
            upper_M_pos @ (alpha_u * b + beta_u)
            + upper_M_neg @ (alpha_l * b + beta_l)
            + upper_p
        )

        return new_lower_M, new_lower_p, new_upper_M, new_upper_p

    @staticmethod
    def _compose_post_activation_bound(
        pre: AffineBound,
        alpha_l: Array,
        beta_l: Array,
        alpha_u: Array,
        beta_u: Array,
    ) -> AffineBound:
        """
        Construct diagnostic affine bounds for f^(k) from bounds for s^(k).

        These post-activation bounds are returned through LayerBound for API
        compatibility and inspection.  They are not used to construct later
        relaxations; later pre-activation bounds are always obtained by fresh
        backward substitution.
        """
        alpha_l_pos = positive_part(alpha_l)
        alpha_l_neg = negative_part(alpha_l)
        alpha_u_pos = positive_part(alpha_u)
        alpha_u_neg = negative_part(alpha_u)

        return AffineBound(
            lower_A=(
                alpha_l_pos[:, None] * pre.lower_A
                + alpha_l_neg[:, None] * pre.upper_A
            ),
            lower_c=(
                alpha_l_pos * pre.lower_c
                + alpha_l_neg * pre.upper_c
                + beta_l
            ),
            upper_A=(
                alpha_u_pos[:, None] * pre.upper_A
                + alpha_u_neg[:, None] * pre.lower_A
            ),
            upper_c=(
                alpha_u_pos * pre.upper_c
                + alpha_u_neg * pre.lower_c
                + beta_u
            ),
        )

    def _relax_activation(
        self,
        activation_name: str,
        pre_lower: Array,
        pre_upper: Array,
    ) -> Tuple[Array, Array, Array, Array]:
        if activation_name == "linear":
            return self._linear_relaxation(pre_lower, pre_upper)

        relaxation = self.activation_relaxations.get(activation_name)
        if relaxation is None:
            raise ValueError(
                f"No relaxation registered for activation: {activation_name}"
            )
        return relaxation.relax(pre_lower, pre_upper)

    def _build_one_layer_relaxation(
        self,
        network: FullyConnectedNetwork,
        layer_index: int,
        x0: Array,
        eps: float | Array,
        previous_layer_bounds: List[LayerBound],
    ) -> LayerBound:
        """
        Build the pre-activation bound and relaxation for one layer.

        ``layer_index`` is zero-based, so it represents mathematical layer
        k = layer_index + 1.  The method assumes relaxations for every earlier
        layer have already been constructed.
        """
        if len(previous_layer_bounds) != layer_index:
            raise ValueError(
                "Relaxations must be constructed consecutively from the first layer."
            )

        # Start from the exact target pre-activation
        #     s^(k) = W^(k) f^(k-1) + b^(k).
        lower_M = network.weights[layer_index].copy()
        upper_M = network.weights[layer_index].copy()
        lower_p = network.biases[layer_index].copy()
        upper_p = network.biases[layer_index].copy()

        # Back-substitute through layers k-1, ..., 1 using only relaxations
        # that were themselves constructed by backward-only analysis.
        for previous_index in range(layer_index - 1, -1, -1):
            previous = previous_layer_bounds[previous_index]
            lower_M, lower_p, upper_M, upper_p = self._backward_one_layer(
                lower_M=lower_M,
                lower_p=lower_p,
                upper_M=upper_M,
                upper_p=upper_p,
                W=network.weights[previous_index],
                b=network.biases[previous_index],
                alpha_l=previous.alpha_lower,
                beta_l=previous.beta_lower,
                alpha_u=previous.alpha_upper,
                beta_u=previous.beta_upper,
            )

        pre = AffineBound(
            lower_A=lower_M,
            lower_c=lower_p,
            upper_A=upper_M,
            upper_c=upper_p,
        )
        pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps)
        pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps)

        alpha_l, beta_l, alpha_u, beta_u = self._relax_activation(
            network.activations[layer_index],
            pre_lower,
            pre_upper,
        )

        post = self._compose_post_activation_bound(
            pre=pre,
            alpha_l=alpha_l,
            beta_l=beta_l,
            alpha_u=alpha_u,
            beta_u=beta_u,
        )
        post_lower = affine_min(post.lower_A, post.lower_c, x0, eps)
        post_upper = affine_max(post.upper_A, post.upper_c, x0, eps)

        return LayerBound(
            pre_affine=pre,
            pre_lower=pre_lower,
            pre_upper=pre_upper,
            alpha_lower=alpha_l,
            beta_lower=beta_l,
            alpha_upper=alpha_u,
            beta_upper=beta_u,
            post_affine=post,
            post_lower=post_lower,
            post_upper=post_upper,
        )

    @staticmethod
    def _validate_input(
        network: FullyConnectedNetwork,
        x0: Array,
        eps: float | Array,
    ) -> Tuple[Array, Array]:
        x0_array = np.asarray(x0, dtype=float)
        if x0_array.ndim != 1:
            raise ValueError("x0 must be a vector.")
        if x0_array.shape[0] != network.input_dim:
            raise ValueError(
                f"x0 dimension {x0_array.shape[0]} does not match "
                f"network input dimension {network.input_dim}."
            )

        eps_array = np.asarray(eps, dtype=float)
        try:
            eps_broadcast = np.broadcast_to(eps_array, x0_array.shape)
        except ValueError as exc:
            raise ValueError("eps must be scalar or broadcastable to x0.") from exc
        if np.any(eps_broadcast < 0.0):
            raise ValueError("eps must be nonnegative.")

        return x0_array, eps_array

    @staticmethod
    def _initialize_output_specification(
        output_dim: int,
        output_lower_M: Array | None,
        output_lower_p: Array | None,
        output_upper_M: Array | None,
        output_upper_p: Array | None,
    ) -> Tuple[Array, Array, Array, Array]:
        lower_M = (
            np.eye(output_dim)
            if output_lower_M is None
            else np.asarray(output_lower_M, dtype=float)
        )
        upper_M = (
            np.eye(output_dim)
            if output_upper_M is None
            else np.asarray(output_upper_M, dtype=float)
        )

        if lower_M.ndim != 2 or upper_M.ndim != 2:
            raise ValueError("Output specification matrices must be two-dimensional.")
        if lower_M.shape[1] != output_dim or upper_M.shape[1] != output_dim:
            raise ValueError(
                "Output specification matrices must have one column per network output."
            )
        if lower_M.shape[0] != upper_M.shape[0]:
            raise ValueError(
                "Lower and upper output specifications must have the same number of rows."
            )

        spec_dim = lower_M.shape[0]
        lower_p = (
            np.zeros(spec_dim)
            if output_lower_p is None
            else np.asarray(output_lower_p, dtype=float)
        )
        upper_p = (
            np.zeros(spec_dim)
            if output_upper_p is None
            else np.asarray(output_upper_p, dtype=float)
        )

        if lower_p.ndim != 1 or upper_p.ndim != 1:
            raise ValueError("Output specification offsets must be vectors.")
        if lower_p.shape[0] != spec_dim or upper_p.shape[0] != spec_dim:
            raise ValueError("Output specification vectors must match their matrices.")

        return lower_M, lower_p, upper_M, upper_p

    def bound(
        self,
        network: FullyConnectedNetwork,
        x0: Array,
        eps: float | Array,
        output_lower_M: Array | None = None,
        output_lower_p: Array | None = None,
        output_upper_M: Array | None = None,
        output_upper_p: Array | None = None,
    ) -> Tuple[AffineBound, Array, Array, List[LayerBound]]:
        """
        Compute backward-only CROWN bounds over [x0-eps, x0+eps].

        The return value matches LiRPAForward.bound and LiRPABackward.bound:

            final_affine_bound, numerical_lower, numerical_upper,
            per_layer_bounds

        By default the identity output specification bounds f^(L) itself.
        Custom lower/upper output specifications are supported in the same form
        as LiRPABackward.bound.
        """
        x0_array, eps_array = self._validate_input(network, x0, eps)

        # Recursive Algorithm 3 can be evaluated iteratively because layer k
        # depends only on relaxations 1, ..., k-1.  No forward LiRPA pass occurs.
        layer_bounds: List[LayerBound] = []
        for layer_index in range(len(network.weights)):
            layer_bound = self._build_one_layer_relaxation(
                network=network,
                layer_index=layer_index,
                x0=x0_array,
                eps=eps_array,
                previous_layer_bounds=layer_bounds,
            )
            layer_bounds.append(layer_bound)

        output_dim = network.weights[-1].shape[0]
        lower_M, lower_p, upper_M, upper_p = self._initialize_output_specification(
            output_dim=output_dim,
            output_lower_M=output_lower_M,
            output_lower_p=output_lower_p,
            output_upper_M=output_upper_M,
            output_upper_p=output_upper_p,
        )

        # With all activation relaxations available, perform the final backward
        # pass for the requested output specification.
        for layer_index in range(len(network.weights) - 1, -1, -1):
            layer_bound = layer_bounds[layer_index]
            lower_M, lower_p, upper_M, upper_p = self._backward_one_layer(
                lower_M=lower_M,
                lower_p=lower_p,
                upper_M=upper_M,
                upper_p=upper_p,
                W=network.weights[layer_index],
                b=network.biases[layer_index],
                alpha_l=layer_bound.alpha_lower,
                beta_l=layer_bound.beta_lower,
                alpha_u=layer_bound.alpha_upper,
                beta_u=layer_bound.beta_upper,
            )

        final = AffineBound(
            lower_A=lower_M,
            lower_c=lower_p,
            upper_A=upper_M,
            upper_c=upper_p,
        )
        final_lower = affine_min(final.lower_A, final.lower_c, x0_array, eps_array)
        final_upper = affine_max(final.upper_A, final.upper_c, x0_array, eps_array)
        return final, final_lower, final_upper, layer_bounds

def make_xor_network_from_note() -> FullyConnectedNetwork:
    """
    XOR network from the uploaded note.

    Hidden layer: 2 ReLU neurons
    Output layer: 1 Sigmoid neuron
    """
    W1 = np.array(
        [
            [2.1247, 2.1267],
            [-2.1237, -2.1235],
        ],
        dtype=float,
    )
    b1 = np.array([-2.1259, 2.1234], dtype=float)

    W2 = np.array([[-3.6788, -3.6766]], dtype=float)
    b2 = np.array([3.5451], dtype=float)

    return FullyConnectedNetwork(
        weights=[W1, W2],
        biases=[b1, b2],
        activations=["relu", "sigmoid"],
    )


def xor_expected_label(x: Array) -> int:
    x = np.asarray(x, dtype=float)
    return int(round(float(x[0]))) ^ int(round(float(x[1])))


def run_xor_demo(eps: float = 0.02) -> None:
    network = make_xor_network_from_note()

    forward_verifier = LiRPAForward()
    forward_backward_verifier = LiRPABackward(forward_verifier)
    backward_only_verifier = LiRPABackwardOnly()

    points = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]

    print("XOR network point predictions and LiRPA-certified output bounds")
    print(f"Perturbation: L_inf epsilon = {eps}")
    print()

    all_certified_forward = True
    all_certified_forward_backward = True
    all_certified_backward_only = True

    for x0 in points:
        y = network.forward(x0)

        _, fwd_lb, fwd_ub, _ = forward_verifier.bound(
            network, x0, eps
        )
        _, fb_lb, fb_ub, _ = forward_backward_verifier.bound(
            network, x0, eps
        )
        _, bo_lb, bo_ub, _ = backward_only_verifier.bound(
            network, x0, eps
        )

        expected = xor_expected_label(x0)

        if expected == 1:
            fwd_certified = bool(fwd_lb[0] > 0.5)
            fb_certified = bool(fb_lb[0] > 0.5)
            bo_certified = bool(bo_lb[0] > 0.5)
            condition = "lower bound > 0.5"
        else:
            fwd_certified = bool(fwd_ub[0] < 0.5)
            fb_certified = bool(fb_ub[0] < 0.5)
            bo_certified = bool(bo_ub[0] < 0.5)
            condition = "upper bound < 0.5"

        all_certified_forward &= fwd_certified
        all_certified_forward_backward &= fb_certified
        all_certified_backward_only &= bo_certified

        print(
            f"x0={x0.tolist()}, expected={expected}, "
            f"network_output={y[0]:.6f}"
        )
        print(
            f"  forward          "
            f"bound=[{fwd_lb[0]:.6f}, {fwd_ub[0]:.6f}], "
            f"certified={fwd_certified} ({condition})"
        )
        print(
            f"  forward+backward "
            f"bound=[{fb_lb[0]:.6f}, {fb_ub[0]:.6f}], "
            f"certified={fb_certified} ({condition})"
        )
        print(
            f"  backward-only    "
            f"bound=[{bo_lb[0]:.6f}, {bo_ub[0]:.6f}], "
            f"certified={bo_certified} ({condition})"
        )
        print()

    mode_results = [
        ("Forward mode", all_certified_forward),
        ("Forward+backward mode", all_certified_forward_backward),
        ("Backward-only mode", all_certified_backward_only),
    ]

    for mode_name, all_certified in mode_results:
        if all_certified:
            print(
                f"{mode_name} certifies all four XOR corner "
                f"classifications for this epsilon."
            )
        else:
            print(
                f"{mode_name} does not certify at least one XOR corner "
                f"classification for this epsilon."
            )


def _self_test_relaxations() -> None:
    """Basic sanity checks that sampled points satisfy the relaxations."""
    rng = np.random.default_rng(0)
    for relaxation, fn in [
        (ReLURelaxation(), ReLURelaxation.relu),
        (SigmoidRelaxation(), SigmoidRelaxation.sigma),
    ]:
        for _ in range(200):
            a, b = sorted(rng.uniform(-5.0, 5.0, size=2))
            if abs(a - b) < 1e-8:
                b = a + 1e-6

            l = np.array([a])
            u = np.array([b])
            alpha_l, beta_l, alpha_u, beta_u = relaxation.relax(l, u)

            xs = np.linspace(a, b, 201)
            ys = fn(xs)
            lhs = alpha_l[0] * xs + beta_l[0]
            rhs = alpha_u[0] * xs + beta_u[0]

            if not np.all(lhs <= ys + 1e-8):
                raise AssertionError(f"{relaxation.name} lower relaxation failed on interval [{a}, {b}].")
            if not np.all(ys <= rhs + 1e-8):
                raise AssertionError(f"{relaxation.name} upper relaxation failed on interval [{a}, {b}].")


if __name__ == "__main__":
    _self_test_relaxations()
    run_xor_demo(eps=0.02)
