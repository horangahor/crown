"""
Iterative backward-only CROWN/LiRPA construction for fully-connected networks.

This module follows the iterative algorithm described in the note:
for each layer k, build a pre-activation affine bound for s^(k) by
back-substituting through layers k-1, ..., 1 using already-computed
relaxation parameters, concretize [s_lower^(k), s_upper^(k)], then build
activation relaxations for layer k.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from lirpa_backward_only import (
    ActivationRelaxation,
    AffineBound,
    Array,
    FullyConnectedNetwork,
    LiRPABackwardOnly,
    LayerBound,
    ReLURelaxation,
    SigmoidRelaxation,
    affine_max,
    affine_min,
    negative_part,
    positive_part,
    make_xor_network_from_note,
    xor_expected_label,
)


class LiRPABackwardOnlyIteration:
    """
    Iterative backward-only CROWN bound propagation.

    The implementation is intentionally explicit about the two loops:
    1) Outer loop over k = 1..L to construct per-layer relaxations.
    2) Inner loop over r = k-1..1 to back-substitute to the input.

    This avoids recursive control flow while preserving the exact same
    computation order and complexity as recursive backward-only CROWN.
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
        del upper
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
        lower_M_pos = positive_part(lower_M)
        lower_M_neg = negative_part(lower_M)
        upper_M_pos = positive_part(upper_M)
        upper_M_neg = negative_part(upper_M)

        lower_s_coeff = lower_M_pos * alpha_l[None, :] + lower_M_neg * alpha_u[None, :]
        new_lower_M = lower_s_coeff @ W
        new_lower_p = (
            lower_M_pos @ (alpha_l * b + beta_l)
            + lower_M_neg @ (alpha_u * b + beta_u)
            + lower_p
        )

        upper_s_coeff = upper_M_pos * alpha_u[None, :] + upper_M_neg * alpha_l[None, :]
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
        alpha_l_pos = positive_part(alpha_l)
        alpha_l_neg = negative_part(alpha_l)
        alpha_u_pos = positive_part(alpha_u)
        alpha_u_neg = negative_part(alpha_u)

        return AffineBound(
            lower_A=alpha_l_pos[:, None] * pre.lower_A + alpha_l_neg[:, None] * pre.upper_A,
            lower_c=alpha_l_pos * pre.lower_c + alpha_l_neg * pre.upper_c + beta_l,
            upper_A=alpha_u_pos[:, None] * pre.upper_A + alpha_u_neg[:, None] * pre.lower_A,
            upper_c=alpha_u_pos * pre.upper_c + alpha_u_neg * pre.lower_c + beta_u,
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
            raise ValueError(f"No relaxation registered for activation: {activation_name}")
        return relaxation.relax(pre_lower, pre_upper)

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
        lower_M = np.eye(output_dim) if output_lower_M is None else np.asarray(output_lower_M, dtype=float)
        upper_M = np.eye(output_dim) if output_upper_M is None else np.asarray(output_upper_M, dtype=float)

        if lower_M.ndim != 2 or upper_M.ndim != 2:
            raise ValueError("Output specification matrices must be two-dimensional.")
        if lower_M.shape[1] != output_dim or upper_M.shape[1] != output_dim:
            raise ValueError("Output specification matrices must have one column per network output.")
        if lower_M.shape[0] != upper_M.shape[0]:
            raise ValueError("Lower and upper output specifications must have the same number of rows.")

        spec_dim = lower_M.shape[0]
        lower_p = np.zeros(spec_dim) if output_lower_p is None else np.asarray(output_lower_p, dtype=float)
        upper_p = np.zeros(spec_dim) if output_upper_p is None else np.asarray(output_upper_p, dtype=float)

        if lower_p.ndim != 1 or upper_p.ndim != 1:
            raise ValueError("Output specification offsets must be vectors.")
        if lower_p.shape[0] != spec_dim or upper_p.shape[0] != spec_dim:
            raise ValueError("Output specification vectors must match their matrices.")

        return lower_M, lower_p, upper_M, upper_p

    @staticmethod
    def _print_layer_debug_info(layer_index: int, layer_bound: LayerBound) -> None:
        """Print concise per-layer pre-activation bounds and relaxation parameters."""
        k = layer_index + 1
        print(f"[debug] layer {k}")
        print(f"  pre_lower={np.array2string(layer_bound.pre_lower, precision=6)}")
        print(f"  pre_upper={np.array2string(layer_bound.pre_upper, precision=6)}")
        print(f"  alpha_lower={np.array2string(layer_bound.alpha_lower, precision=6)}")
        print(f"  beta_lower={np.array2string(layer_bound.beta_lower, precision=6)}")
        print(f"  alpha_upper={np.array2string(layer_bound.alpha_upper, precision=6)}")
        print(f"  beta_upper={np.array2string(layer_bound.beta_upper, precision=6)}")

    def _build_layer_relaxations_iterative(
        self,
        network: FullyConnectedNetwork,
        x0: Array,
        eps: float | Array,
        debug: bool = False,
    ) -> List[LayerBound]:
        """Construct all per-layer pre-activation bounds and relaxations by iteration."""
        layer_bounds: List[LayerBound] = []
        total_layers = len(network.weights)

        for k in range(total_layers):
            # C_k^(k-1), d_k^(k-1) initialization from s^(k) = W^(k) f^(k-1) + b^(k).
            lower_C = network.weights[k].copy()
            upper_C = network.weights[k].copy()
            lower_d = network.biases[k].copy()
            upper_d = network.biases[k].copy()

            # r = k-1 down to 1 (0-based: k-1 down to 0).
            for r in range(k - 1, -1, -1):
                prev = layer_bounds[r]
                W_r = network.weights[r]
                b_r = network.biases[r]

                lower_C_pos = positive_part(lower_C)
                lower_C_neg = negative_part(lower_C)
                lower_C = (
                    lower_C_pos * prev.alpha_lower[None, :]
                    + lower_C_neg * prev.alpha_upper[None, :]
                ) @ W_r
                lower_d = (
                    lower_d
                    + lower_C_pos @ (prev.alpha_lower * b_r + prev.beta_lower)
                    + lower_C_neg @ (prev.alpha_upper * b_r + prev.beta_upper)
                )

                upper_C_pos = positive_part(upper_C)
                upper_C_neg = negative_part(upper_C)
                upper_C = (
                    upper_C_pos * prev.alpha_upper[None, :]
                    + upper_C_neg * prev.alpha_lower[None, :]
                ) @ W_r
                upper_d = (
                    upper_d
                    + upper_C_pos @ (prev.alpha_upper * b_r + prev.beta_upper)
                    + upper_C_neg @ (prev.alpha_lower * b_r + prev.beta_lower)
                )

            pre = AffineBound(
                lower_A=lower_C,
                lower_c=lower_d,
                upper_A=upper_C,
                upper_c=upper_d,
            )
            pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps)
            pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps)

            alpha_l, beta_l, alpha_u, beta_u = self._relax_activation(
                activation_name=network.activations[k],
                pre_lower=pre_lower,
                pre_upper=pre_upper,
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

            if debug:
                self._print_layer_debug_info(k, layer_bounds[-1])

        return layer_bounds

    def bound(
        self,
        network: FullyConnectedNetwork,
        x0: Array,
        eps: float | Array,
        output_lower_M: Array | None = None,
        output_lower_p: Array | None = None,
        output_upper_M: Array | None = None,
        output_upper_p: Array | None = None,
        debug: bool = False,
    ) -> Tuple[AffineBound, Array, Array, List[LayerBound]]:
        """
        Compute iterative backward-only CROWN bounds over [x0-eps, x0+eps].

        Returns:
            final_affine_bound, numerical_lower, numerical_upper, per_layer_bounds
        """
        x0_array, eps_array = self._validate_input(network, x0, eps)

        layer_bounds = self._build_layer_relaxations_iterative(
            network=network,
            x0=x0_array,
            eps=eps_array,
            debug=debug,
        )

        output_dim = network.weights[-1].shape[0]
        lower_M, lower_p, upper_M, upper_p = self._initialize_output_specification(
            output_dim=output_dim,
            output_lower_M=output_lower_M,
            output_lower_p=output_lower_p,
            output_upper_M=output_upper_M,
            output_upper_p=output_upper_p,
        )

        for layer_index in range(len(network.weights) - 1, -1, -1):
            layer = layer_bounds[layer_index]
            lower_M, lower_p, upper_M, upper_p = self._backward_one_layer(
                lower_M=lower_M,
                lower_p=lower_p,
                upper_M=upper_M,
                upper_p=upper_p,
                W=network.weights[layer_index],
                b=network.biases[layer_index],
                alpha_l=layer.alpha_lower,
                beta_l=layer.beta_lower,
                alpha_u=layer.alpha_upper,
                beta_u=layer.beta_upper,
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


def run_xor_demo_iterative(eps: float = 0.02, debug: bool = False) -> None:
    """Quick demo for iterative backward-only bounds on the XOR toy network."""
    network = make_xor_network_from_note()
    verifier = LiRPABackwardOnlyIteration()

    points = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]

    print("XOR iterative backward-only CROWN bounds")
    print(f"Perturbation: L_inf epsilon = {eps}")
    print()

    all_certified = True
    for x0 in points:
        y = network.forward(x0)
        _, lb, ub, _ = verifier.bound(network, x0, eps, debug=debug)

        expected = xor_expected_label(x0)
        if expected == 1:
            certified = bool(lb[0] > 0.5)
            condition = "lower bound > 0.5"
        else:
            certified = bool(ub[0] < 0.5)
            condition = "upper bound < 0.5"

        all_certified &= certified

        print(f"x0={x0.tolist()}, expected={expected}, network_output={y[0]:.6f}")
        print(f"  iterative backward-only bound=[{lb[0]:.6f}, {ub[0]:.6f}], certified={certified} ({condition})")
        print()

    if all_certified:
        print("Iterative backward-only mode certifies all four XOR corner classifications for this epsilon.")
    else:
        print("Iterative backward-only mode does not certify at least one XOR corner classification for this epsilon.")


def compare_recursive_and_iterative(
    eps: float = 0.02,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> None:
    """
    Compare recursive and iterative backward-only implementations on XOR points.

    Raises AssertionError if any output bound or layer-wise pre-activation
    interval differs beyond tolerance.
    """
    network = make_xor_network_from_note()
    recursive_verifier = LiRPABackwardOnly()
    iterative_verifier = LiRPABackwardOnlyIteration()

    points = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]

    print("Comparing recursive vs iterative backward-only bounds")
    print(f"Tolerance: atol={atol}, rtol={rtol}")
    print()

    for x0 in points:
        _, rec_lb, rec_ub, rec_layers = recursive_verifier.bound(network, x0, eps)
        _, itr_lb, itr_ub, itr_layers = iterative_verifier.bound(network, x0, eps)

        if not np.allclose(rec_lb, itr_lb, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Lower bound mismatch at x0={x0.tolist()}: "
                f"recursive={rec_lb}, iterative={itr_lb}"
            )
        if not np.allclose(rec_ub, itr_ub, atol=atol, rtol=rtol):
            raise AssertionError(
                f"Upper bound mismatch at x0={x0.tolist()}: "
                f"recursive={rec_ub}, iterative={itr_ub}"
            )

        if len(rec_layers) != len(itr_layers):
            raise AssertionError("Layer count mismatch between recursive and iterative results.")

        for layer_index, (rec_layer, itr_layer) in enumerate(zip(rec_layers, itr_layers), start=1):
            if not np.allclose(rec_layer.pre_lower, itr_layer.pre_lower, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"pre_lower mismatch at layer {layer_index}, x0={x0.tolist()}"
                )
            if not np.allclose(rec_layer.pre_upper, itr_layer.pre_upper, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"pre_upper mismatch at layer {layer_index}, x0={x0.tolist()}"
                )
            if not np.allclose(rec_layer.alpha_lower, itr_layer.alpha_lower, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"alpha_lower mismatch at layer {layer_index}, x0={x0.tolist()}"
                )
            if not np.allclose(rec_layer.beta_lower, itr_layer.beta_lower, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"beta_lower mismatch at layer {layer_index}, x0={x0.tolist()}"
                )
            if not np.allclose(rec_layer.alpha_upper, itr_layer.alpha_upper, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"alpha_upper mismatch at layer {layer_index}, x0={x0.tolist()}"
                )
            if not np.allclose(rec_layer.beta_upper, itr_layer.beta_upper, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"beta_upper mismatch at layer {layer_index}, x0={x0.tolist()}"
                )

        print(f"x0={x0.tolist()}: OK")

    print()
    print("Recursive and iterative backward-only results match for all tested points.")


if __name__ == "__main__":
    compare_recursive_and_iterative(eps=0.02)
    print()
    run_xor_demo_iterative(eps=0.02, debug=False)
