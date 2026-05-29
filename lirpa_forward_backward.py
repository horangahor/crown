"""
Forward and backward LiRPA for small fully connected neural networks.

Assumptions
-----------
1. Each layer is fully connected:
       s^(l) = W^(l) f^(l-1) + b^(l)
2. Hidden activations are ReLU.
3. The last layer may use Sigmoid.
4. The perturbation set is an L_infinity ball:
       x_i in [x0_i - eps, x0_i + eps]

The code contains:

- ForwardLiRPA:
    Numeric interval propagation from input to output.

- BackwardLiRPA:
    Symbolic backward bound propagation for the last pre-sigmoid logit.
    This is useful for binary classification verification because
        sigmoid(z) >= 0.5  iff  z >= 0.

The activation abstraction is modular. To add another activation, subclass
Activation and implement value(), interval(), and linear_relaxation().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np


Array = np.ndarray


@dataclass
class Interval:
    """Elementwise lower and upper bounds."""
    lower: Array
    upper: Array

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(self.lower > self.upper):
            raise ValueError("invalid interval: lower > upper")

    def copy(self) -> "Interval":
        return Interval(self.lower.copy(), self.upper.copy())


@dataclass
class LinearRelaxation:
    """
    Elementwise affine relaxation of activation sigma(s):

        lower_slope * s + lower_bias <= sigma(s)
        sigma(s) <= upper_slope * s + upper_bias
    """
    lower_slope: Array
    lower_bias: Array
    upper_slope: Array
    upper_bias: Array


class Activation:
    name: str = "activation"

    def value(self, x: Array) -> Array:
        raise NotImplementedError

    def interval(self, pre: Interval) -> Interval:
        raise NotImplementedError

    def linear_relaxation(self, pre: Interval) -> LinearRelaxation:
        raise NotImplementedError


class Identity(Activation):
    name = "identity"

    def value(self, x: Array) -> Array:
        return np.asarray(x, dtype=float)

    def interval(self, pre: Interval) -> Interval:
        return pre.copy()

    def linear_relaxation(self, pre: Interval) -> LinearRelaxation:
        z = np.zeros_like(pre.lower)
        o = np.ones_like(pre.lower)
        return LinearRelaxation(o, z, o, z)


class ReLU(Activation):
    name = "relu"

    def value(self, x: Array) -> Array:
        return np.maximum(0.0, x)

    def interval(self, pre: Interval) -> Interval:
        # Exact interval image of ReLU over [l,u].
        return Interval(np.maximum(0.0, pre.lower), np.maximum(0.0, pre.upper))

    def linear_relaxation(self, pre: Interval) -> LinearRelaxation:
        """
        ReLU linear relaxation.

        Cases:
        1. 0 <= l <= u:
             lower = s, upper = s
        2. l <= u <= 0:
             lower = 0, upper = 0
        3. l <= 0 <= u and |l| <= |u|:
             lower = 0
             upper = (u/(u-l))s - (u*l)/(u-l)
        4. l <= 0 <= u and |l| > |u|:
             lower = s
             upper = (u/(u-l))s - (u*l)/(u-l)
        """
        l = pre.lower
        u = pre.upper

        lower_slope = np.zeros_like(l)
        lower_bias = np.zeros_like(l)
        upper_slope = np.zeros_like(l)
        upper_bias = np.zeros_like(l)

        pos = l >= 0.0
        neg = u <= 0.0
        cross = ~(pos | neg)

        # Fully positive.
        lower_slope[pos] = 1.0
        upper_slope[pos] = 1.0

        # Fully negative remains zero.

        # Crossing zero.
        denom = u[cross] - l[cross]
        if denom.size > 0:
            upper_slope[cross] = u[cross] / denom
            upper_bias[cross] = -u[cross] * l[cross] / denom

            # Lower-bound heuristic:
            # ReLU(s) >= 0 and ReLU(s) >= s are both valid.
            # Use s when |l| > |u|, otherwise use 0.
            lower_slope[cross] = (np.abs(l[cross]) > np.abs(u[cross])).astype(float)

        return LinearRelaxation(lower_slope, lower_bias, upper_slope, upper_bias)


class Sigmoid(Activation):
    name = "sigmoid"

    def value(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    def derivative(self, x: Array) -> Array:
        y = self.value(x)
        return y * (1.0 - y)

    def interval(self, pre: Interval) -> Interval:
        # Exact interval image because sigmoid is monotone increasing.
        return Interval(self.value(pre.lower), self.value(pre.upper))

    def linear_relaxation(self, pre: Interval) -> LinearRelaxation:
        """
        Basic modular sigmoid relaxation.

        Forward interval propagation uses interval(), which is exact for sigmoid.
        Backward binary verification in this file bounds the pre-sigmoid logit,
        so it also does not need this method.

        This method is included for future extensions where the final sigmoid
        itself is handled symbolically.
        """
        l = pre.lower
        u = pre.upper
        mid = (l + u) / 2.0

        sig_l = self.value(l)
        sig_u = self.value(u)
        width = np.maximum(u - l, 1e-12)

        secant_slope = (sig_u - sig_l) / width
        secant_bias = sig_u - secant_slope * u

        tangent_slope = self.derivative(mid)
        tangent_bias = self.value(mid) - tangent_slope * mid

        lower_slope = np.empty_like(l)
        lower_bias = np.empty_like(l)
        upper_slope = np.empty_like(l)
        upper_bias = np.empty_like(l)

        pos = l >= 0
        neg = u <= 0
        cross = ~(pos | neg)

        # On x <= 0, sigmoid is convex: tangent lower, secant upper.
        lower_slope[neg] = tangent_slope[neg]
        lower_bias[neg] = tangent_bias[neg]
        upper_slope[neg] = secant_slope[neg]
        upper_bias[neg] = secant_bias[neg]

        # On x >= 0, sigmoid is concave: secant lower, tangent upper.
        lower_slope[pos] = secant_slope[pos]
        lower_bias[pos] = secant_bias[pos]
        upper_slope[pos] = tangent_slope[pos]
        upper_bias[pos] = tangent_bias[pos]

        # Simple placeholder for crossing-zero intervals.
        # A sharper implementation can replace this with the binary-search
        # tangent construction described in LiRPA/CROWN-style algorithms.
        lower_slope[cross] = secant_slope[cross]
        lower_bias[cross] = secant_bias[cross]
        upper_slope[cross] = secant_slope[cross]
        upper_bias[cross] = secant_bias[cross]

        return LinearRelaxation(lower_slope, lower_bias, upper_slope, upper_bias)


@dataclass
class DenseLayer:
    W: Array
    b: Array
    activation: Activation

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        if self.W.ndim != 2:
            raise ValueError("W must be a 2D matrix")
        if self.b.ndim != 1:
            raise ValueError("b must be a 1D vector")
        if self.W.shape[0] != self.b.shape[0]:
            raise ValueError("W.shape[0] must match b.shape[0]")

    def forward(self, x: Array) -> Array:
        return self.activation.value(self.W @ x + self.b)


@dataclass
class LayerBound:
    pre: Interval
    post: Interval
    activation_name: str


@dataclass
class AffineBound:
    """
    Symbolic affine lower/upper bounds over the input x:

        lower_coeff @ x + lower_bias <= target(x)
        target(x) <= upper_coeff @ x + upper_bias
    """
    lower_coeff: Array
    lower_bias: float
    upper_coeff: Array
    upper_bias: float


class FullyConnectedNetwork:
    def __init__(self, layers: Sequence[DenseLayer]):
        if not layers:
            raise ValueError("network must have at least one layer")
        self.layers = list(layers)

    def forward(self, x: Array) -> Array:
        h = np.asarray(x, dtype=float)
        for layer in self.layers:
            h = layer.forward(h)
        return h


def affine_interval(W: Array, b: Array, inp: Interval) -> Interval:
    """
    Interval bound for s = W x + b.

        W^+ L + W^- U + b <= W x + b <= W^+ U + W^- L + b
    """
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float)

    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)

    lower = W_pos @ inp.lower + W_neg @ inp.upper + b
    upper = W_pos @ inp.upper + W_neg @ inp.lower + b
    return Interval(lower, upper)


def evaluate_affine_lower(coeff: Array, bias: float, inp: Interval) -> float:
    """min_x coeff @ x + bias over x in [L,U]."""
    coeff = np.asarray(coeff, dtype=float)
    c_pos = np.maximum(coeff, 0.0)
    c_neg = np.minimum(coeff, 0.0)
    return float(c_pos @ inp.lower + c_neg @ inp.upper + bias)


def evaluate_affine_upper(coeff: Array, bias: float, inp: Interval) -> float:
    """max_x coeff @ x + bias over x in [L,U]."""
    coeff = np.asarray(coeff, dtype=float)
    c_pos = np.maximum(coeff, 0.0)
    c_neg = np.minimum(coeff, 0.0)
    return float(c_pos @ inp.upper + c_neg @ inp.lower + bias)


class ForwardLiRPA:
    """
    Forward-mode LiRPA / interval bound propagation.
    """

    def __init__(self, network: FullyConnectedNetwork):
        self.network = network

    def bound(self, x0: Array, epsilon: float) -> Tuple[Interval, List[LayerBound]]:
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        x0 = np.asarray(x0, dtype=float)
        current = Interval(x0 - epsilon, x0 + epsilon)

        trace: List[LayerBound] = []
        for layer in self.network.layers:
            pre = affine_interval(layer.W, layer.b, current)
            post = layer.activation.interval(pre)
            trace.append(LayerBound(pre=pre, post=post, activation_name=layer.activation.name))
            current = post

        return current, trace

    def verify_binary_classification(
        self,
        x0: Array,
        epsilon: float,
        expected_label: int,
        threshold: float = 0.5,
    ) -> Tuple[bool, Interval, List[LayerBound]]:
        out, trace = self.bound(x0, epsilon)
        if out.lower.size != 1:
            raise ValueError("binary verification expects one output")

        if expected_label == 0:
            ok = bool(out.upper[0] < threshold)
        elif expected_label == 1:
            ok = bool(out.lower[0] >= threshold)
        else:
            raise ValueError("expected_label must be 0 or 1")

        return ok, out, trace


class BackwardLiRPA:
    """
    Backward-mode symbolic LiRPA for the last pre-sigmoid logit.

    For binary networks ending in:
        z = W_last h + b_last
        y = sigmoid(z)

    this class computes symbolic affine bounds for z over the input box.
    Then sigmoid monotonicity gives output bounds:
        sigmoid(z_lower) <= y <= sigmoid(z_upper)

    This avoids the need to relax the final sigmoid when the property is
    binary classification with threshold 0.5.
    """

    def __init__(self, network: FullyConnectedNetwork):
        self.network = network
        self.forward_lirpa = ForwardLiRPA(network)

    @staticmethod
    def _backward_one_activation(
        coeff: Array,
        bias: float,
        relax: LinearRelaxation,
        want: str,
    ) -> Tuple[Array, float]:
        """
        Backpropagate an affine expression through an activation relaxation.

        Suppose expr = coeff @ f + bias, where f = activation(s).

        For a lower bound:
            coeff_i >= 0 uses lower relaxation of f_i
            coeff_i <  0 uses upper relaxation of f_i

        For an upper bound:
            coeff_i >= 0 uses upper relaxation of f_i
            coeff_i <  0 uses lower relaxation of f_i
        """
        coeff = np.asarray(coeff, dtype=float)
        c_pos = np.maximum(coeff, 0.0)
        c_neg = np.minimum(coeff, 0.0)

        if want == "lower":
            new_coeff = c_pos * relax.lower_slope + c_neg * relax.upper_slope
            new_bias = (
                bias
                + float(c_pos @ relax.lower_bias)
                + float(c_neg @ relax.upper_bias)
            )
        elif want == "upper":
            new_coeff = c_pos * relax.upper_slope + c_neg * relax.lower_slope
            new_bias = (
                bias
                + float(c_pos @ relax.upper_bias)
                + float(c_neg @ relax.lower_bias)
            )
        else:
            raise ValueError("want must be 'lower' or 'upper'")

        return new_coeff, new_bias

    @staticmethod
    def _backward_one_dense(
        coeff_on_pre: Array,
        bias: float,
        layer: DenseLayer,
    ) -> Tuple[Array, float]:
        """
        Substitute s = W h + b into coeff_on_pre @ s + bias.
        """
        new_coeff = coeff_on_pre @ layer.W
        new_bias = bias + float(coeff_on_pre @ layer.b)
        return new_coeff, new_bias

    def bound_logit(
        self,
        x0: Array,
        epsilon: float,
        output_index: int = 0,
    ) -> Tuple[Interval, AffineBound, List[LayerBound]]:
        """
        Bound the final pre-sigmoid logit z_j.

        The last network layer must be a DenseLayer with Sigmoid activation.
        We start from z_j = W_last[j] @ f^(L-1) + b_last[j], then propagate
        backward through hidden ReLU layers.
        """
        if epsilon < 0:
            raise ValueError("epsilon must be nonnegative")

        x0 = np.asarray(x0, dtype=float)
        input_box = Interval(x0 - epsilon, x0 + epsilon)

        # Forward pass supplies pre-activation intervals for ReLU relaxations.
        _, trace = self.forward_lirpa.bound(x0, epsilon)

        layers = self.network.layers
        last = layers[-1]
        if not isinstance(last.activation, Sigmoid):
            raise ValueError("bound_logit currently expects the last activation to be Sigmoid")
        if output_index < 0 or output_index >= last.W.shape[0]:
            raise ValueError("invalid output_index")

        # Start from the selected pre-sigmoid logit:
        # z_j = W_last[j] @ f^(L-1) + b_last[j].
        lower_coeff = last.W[output_index].copy()
        upper_coeff = last.W[output_index].copy()
        lower_bias = float(last.b[output_index])
        upper_bias = float(last.b[output_index])

        # Propagate backward through all previous layers.
        # If layers = [hidden_1, ..., hidden_k, sigmoid_output],
        # then reverse over hidden_k, ..., hidden_1.
        for layer_index in range(len(layers) - 2, -1, -1):
            layer = layers[layer_index]
            layer_trace = trace[layer_index]
            relax = layer.activation.linear_relaxation(layer_trace.pre)

            # Lower symbolic bound path.
            lower_coeff, lower_bias = self._backward_one_activation(
                lower_coeff, lower_bias, relax, want="lower"
            )
            lower_coeff, lower_bias = self._backward_one_dense(
                lower_coeff, lower_bias, layer
            )

            # Upper symbolic bound path.
            upper_coeff, upper_bias = self._backward_one_activation(
                upper_coeff, upper_bias, relax, want="upper"
            )
            upper_coeff, upper_bias = self._backward_one_dense(
                upper_coeff, upper_bias, layer
            )

        logit_lower = evaluate_affine_lower(lower_coeff, lower_bias, input_box)
        logit_upper = evaluate_affine_upper(upper_coeff, upper_bias, input_box)

        return (
            Interval(np.array([logit_lower]), np.array([logit_upper])),
            AffineBound(lower_coeff, lower_bias, upper_coeff, upper_bias),
            trace,
        )

    def bound_output(
        self,
        x0: Array,
        epsilon: float,
        output_index: int = 0,
    ) -> Tuple[Interval, Interval, AffineBound, List[LayerBound]]:
        """
        Bound both the final pre-sigmoid logit and the final sigmoid output.
        """
        logit_interval, affine, trace = self.bound_logit(x0, epsilon, output_index)
        sigmoid = Sigmoid()
        output_interval = sigmoid.interval(logit_interval)
        return output_interval, logit_interval, affine, trace

    def verify_binary_classification(
        self,
        x0: Array,
        epsilon: float,
        expected_label: int,
        output_index: int = 0,
    ) -> Tuple[bool, Interval, Interval, AffineBound, List[LayerBound]]:
        """
        Binary verification using the pre-sigmoid logit.

        expected_label = 0:
            robust if logit upper < 0
        expected_label = 1:
            robust if logit lower >= 0
        """
        out_interval, logit_interval, affine, trace = self.bound_output(
            x0, epsilon, output_index
        )

        if expected_label == 0:
            ok = bool(logit_interval.upper[0] < 0.0)
        elif expected_label == 1:
            ok = bool(logit_interval.lower[0] >= 0.0)
        else:
            raise ValueError("expected_label must be 0 or 1")

        return ok, out_interval, logit_interval, affine, trace


def make_xor_network() -> FullyConnectedNetwork:
    """
    XOR network from the LiRPA note.
    """
    return FullyConnectedNetwork(
        layers=[
            DenseLayer(
                W=np.array([[2.1247, 2.1267],
                            [-2.1237, -2.1235]]),
                b=np.array([-2.1259, 2.1234]),
                activation=ReLU(),
            ),
            DenseLayer(
                W=np.array([[-3.6788, -3.6766]]),
                b=np.array([3.5451]),
                activation=Sigmoid(),
            ),
        ]
    )


def xor_label(x: Sequence[int]) -> int:
    return int(bool(x[0]) ^ bool(x[1]))


def run_xor_tests() -> None:
    net = make_xor_network()
    fwd = ForwardLiRPA(net)
    bwd = BackwardLiRPA(net)

    tests = [
        # input, epsilon, expected label, expected robust?
        ((0.0, 0.0), 0.100, 0, True),
        ((0.0, 0.0), 0.275, 0, False),
        ((0.0, 1.0), 0.100, 1, True),
        ((1.0, 0.0), 0.100, 1, True),
        ((1.0, 1.0), 0.100, 0, True),
    ]

    print("Forward LiRPA tests")
    for x, eps, label, expected_ok in tests:
        ok, out, _ = fwd.verify_binary_classification(
            np.array(x), eps, expected_label=label
        )
        print(f"x={x}, eps={eps:.3f}, label={label}, "
              f"output=[{out.lower[0]:.6f}, {out.upper[0]:.6f}], "
              f"robust={ok}")
        assert ok == expected_ok

    print("\nBackward LiRPA tests")
    for x, eps, label, expected_ok in tests:
        ok, out, logit, affine, _ = bwd.verify_binary_classification(
            np.array(x), eps, expected_label=label
        )
        print(f"x={x}, eps={eps:.3f}, label={label}, "
              f"logit=[{logit.lower[0]:.6f}, {logit.upper[0]:.6f}], "
              f"output=[{out.lower[0]:.6f}, {out.upper[0]:.6f}], "
              f"robust={ok}")
        assert ok == expected_ok

    print("\nDetailed backward trace for x=(0,0), eps=0.275")
    ok, out, logit, affine, trace = bwd.verify_binary_classification(
        np.array([0.0, 0.0]), 0.275, expected_label=0
    )
    print(f"logit interval  = [{logit.lower[0]:.9f}, {logit.upper[0]:.9f}]")
    print(f"output interval = [{out.lower[0]:.9f}, {out.upper[0]:.9f}]")
    print(f"robust          = {ok}")
    print("symbolic lower bound:")
    print(f"  {affine.lower_coeff} @ x + {affine.lower_bias:.9f}")
    print("symbolic upper bound:")
    print(f"  {affine.upper_coeff} @ x + {affine.upper_bias:.9f}")


if __name__ == "__main__":
    run_xor_tests()
