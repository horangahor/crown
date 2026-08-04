function lirpa_forward_backward_fc_matlab()
% LIRPA_FORWARD_BACKWARD_FC_MATLAB
% A small educational LiRPA forward/backward-mode implementation for
% fully-connected networks with ReLU hidden activations and a Sigmoid output.
%
% The structure follows the accompanying Python implementation as closely as
% practical in MATLAB without using classes.  Bounds are represented by structs:
%
%   bound.lower_A * x + bound.lower_c <= f <= bound.upper_A * x + bound.upper_c
%
% Forward mode propagates affine lower/upper bounds layer by layer.
% Backward mode reuses the activation slopes/intercepts computed by forward
% mode and propagates an output specification back to the input.
%
% Run in MATLAB:
%   lirpa_forward_backward_fc_matlab

    self_test_relaxations();
    run_xor_demo(0.02);
end

%% ------------------------------------------------------------------------
% Demo and network construction
% -------------------------------------------------------------------------

function network = make_xor_network_from_note()
% XOR network from the LiRPA note.
% Hidden layer: 2 ReLU neurons
% Output layer: 1 Sigmoid neuron

    W1 = [ 2.1247,  2.1267;
          -2.1237, -2.1235 ];
    b1 = [-2.1259; 2.1234];

    W2 = [-3.6788, -3.6766];
    b2 = 3.5451;

    network.weights = {W1, W2};
    network.biases = {b1, b2};
    network.activations = {'relu', 'sigmoid'};

    check_network(network);
end

function run_xor_demo(eps_value)
    network = make_xor_network_from_note();

    points = [0 0;
              0 1;
              1 0;
              1 1]';

    fprintf('XOR network point predictions and LiRPA-certified output bounds\n');
    fprintf('Perturbation: L_inf epsilon = %.6f\n\n', eps_value);

    all_certified_forward = true;
    all_certified_backward = true;

    for k = 1:size(points, 2)
        x0 = points(:, k);
        y = network_forward(network, x0);

        [~, fwd_lb, fwd_ub, layer_bounds] = lirpa_forward_bound(network, x0, eps_value);
        [~, bwd_lb, bwd_ub, ~] = lirpa_backward_bound(network, x0, eps_value, layer_bounds);

        expected = xor_expected_label(x0);

        if expected == 1
            fwd_certified = fwd_lb(1) > 0.5;
            bwd_certified = bwd_lb(1) > 0.5;
            condition = 'lower bound > 0.5';
        else
            fwd_certified = fwd_ub(1) < 0.5;
            bwd_certified = bwd_ub(1) < 0.5;
            condition = 'upper bound < 0.5';
        end

        all_certified_forward = all_certified_forward && fwd_certified;
        all_certified_backward = all_certified_backward && bwd_certified;

        fprintf('x0=[%.1f, %.1f], expected=%d, network_output=%.6f\n', ...
            x0(1), x0(2), expected, y(1));
        fprintf('  forward  bound=[%.6f, %.6f], certified=%d (%s)\n', ...
            fwd_lb(1), fwd_ub(1), fwd_certified, condition);
        fprintf('  backward bound=[%.6f, %.6f], certified=%d (%s)\n\n', ...
            bwd_lb(1), bwd_ub(1), bwd_certified, condition);
    end

    if all_certified_forward
        fprintf('Forward mode certifies all four XOR corner classifications for this epsilon.\n');
    else
        fprintf('Forward mode does not certify at least one XOR corner classification for this epsilon.\n');
    end

    if all_certified_backward
        fprintf('Backward mode certifies all four XOR corner classifications for this epsilon.\n');
    else
        fprintf('Backward mode does not certify at least one XOR corner classification for this epsilon.\n');
    end
end

function expected = xor_expected_label(x)
    expected = xor(round(x(1)) ~= 0, round(x(2)) ~= 0);
    expected = double(expected);
end

%% ------------------------------------------------------------------------
% Network evaluation
% -------------------------------------------------------------------------

function y = network_forward(network, x)
    f = x(:);
    for l = 1:numel(network.weights)
        W = network.weights{l};
        b = network.biases{l};
        act = lower(network.activations{l});

        s = W * f + b;
        switch act
            case 'relu'
                f = relu_fn(s);
            case 'sigmoid'
                f = sigmoid_fn(s);
            case 'linear'
                f = s;
            otherwise
                error('Unsupported activation: %s', act);
        end
    end
    y = f;
end

function check_network(network)
    if ~(numel(network.weights) == numel(network.biases) && ...
         numel(network.weights) == numel(network.activations))
        error('weights, biases, and activations must have the same length.');
    end

    for l = 1:numel(network.weights)
        W = network.weights{l};
        b = network.biases{l};

        if ndims(W) ~= 2
            error('W{%d} must be a matrix.', l);
        end
        if size(b, 2) ~= 1
            error('b{%d} must be a column vector.', l);
        end
        if size(W, 1) ~= size(b, 1)
            error('W{%d} rows must match b{%d} length.', l, l);
        end
        if l > 1 && size(network.weights{l-1}, 1) ~= size(W, 2)
            error('Layer dimension mismatch before layer %d.', l);
        end
    end
end

%% ------------------------------------------------------------------------
% Forward-mode LiRPA
% -------------------------------------------------------------------------

function [final_affine, final_lower, final_upper, layer_bounds] = ...
    lirpa_forward_bound(network, x0, eps_value)
% Compute final output bounds for all x in [x0 - eps, x0 + eps].

    x0 = x0(:);
    input_dim = size(network.weights{1}, 2);
    if numel(x0) ~= input_dim
        error('x0 dimension does not match network input dimension.');
    end

    current.lower_A = eye(input_dim);
    current.lower_c = zeros(input_dim, 1);
    current.upper_A = eye(input_dim);
    current.upper_c = zeros(input_dim, 1);

    layer_bounds = cell(numel(network.weights), 1);

    for l = 1:numel(network.weights)
        W = network.weights{l};
        b = network.biases{l};
        act = lower(network.activations{l});

        W_pos = positive_part(W);
        W_neg = negative_part(W);

        % Pre-activation bound for s = W f + b using conditional multiplication.
        pre.lower_A = W_pos * current.lower_A + W_neg * current.upper_A;
        pre.lower_c = W_pos * current.lower_c + W_neg * current.upper_c + b;
        pre.upper_A = W_pos * current.upper_A + W_neg * current.lower_A;
        pre.upper_c = W_pos * current.upper_c + W_neg * current.lower_c + b;

        pre_lower = affine_min(pre.lower_A, pre.lower_c, x0, eps_value);
        pre_upper = affine_max(pre.upper_A, pre.upper_c, x0, eps_value);

        [alpha_l, beta_l, alpha_u, beta_u] = activation_relax(act, pre_lower, pre_upper);

        % ReLU and Sigmoid are monotone with nonnegative relaxation slopes.
        post.lower_A = alpha_l .* pre.lower_A;
        post.lower_c = alpha_l .* pre.lower_c + beta_l;
        post.upper_A = alpha_u .* pre.upper_A;
        post.upper_c = alpha_u .* pre.upper_c + beta_u;

        post_lower = affine_min(post.lower_A, post.lower_c, x0, eps_value);
        post_upper = affine_max(post.upper_A, post.upper_c, x0, eps_value);

        lb.pre_affine = pre;
        lb.pre_lower = pre_lower;
        lb.pre_upper = pre_upper;
        lb.alpha_lower = alpha_l;
        lb.beta_lower = beta_l;
        lb.alpha_upper = alpha_u;
        lb.beta_upper = beta_u;
        lb.post_affine = post;
        lb.post_lower = post_lower;
        lb.post_upper = post_upper;
        layer_bounds{l} = lb;

        current = post;
    end

    final_affine = current;
    final_lower = affine_min(current.lower_A, current.lower_c, x0, eps_value);
    final_upper = affine_max(current.upper_A, current.upper_c, x0, eps_value);
end

%% ------------------------------------------------------------------------
% Backward-mode LiRPA
% -------------------------------------------------------------------------

function [final_affine, final_lower, final_upper, layer_bounds] = ...
    lirpa_backward_bound(network, x0, eps_value, layer_bounds, ...
                         output_lower_M, output_lower_p, output_upper_M, output_upper_p)
% Backward-mode LiRPA bound propagation.
%
% This reuses layer_bounds computed by forward mode.  If layer_bounds is empty
% or omitted, forward mode is first run to compute activation slopes/intercepts.
%
% By default this bounds the network output itself:
%     I f^(L) + 0 <= f^(L) <= I f^(L) + 0.
%
% Custom output specifications can be supplied.  For example, a row vector
% e_y - e_t can be used to bound a logit margin in a multi-output classifier.

    if nargin < 4 || isempty(layer_bounds)
        [~, ~, ~, layer_bounds] = lirpa_forward_bound(network, x0, eps_value);
    end

    output_dim = size(network.weights{end}, 1);

    if nargin < 5 || isempty(output_lower_M)
        lower_M = eye(output_dim);
    else
        lower_M = output_lower_M;
    end

    if nargin < 6 || isempty(output_lower_p)
        lower_p = zeros(size(lower_M, 1), 1);
    else
        lower_p = output_lower_p(:);
    end

    if nargin < 7 || isempty(output_upper_M)
        upper_M = eye(output_dim);
    else
        upper_M = output_upper_M;
    end

    if nargin < 8 || isempty(output_upper_p)
        upper_p = zeros(size(upper_M, 1), 1);
    else
        upper_p = output_upper_p(:);
    end

    if size(lower_M, 2) ~= output_dim || size(upper_M, 2) ~= output_dim
        error('Output specification matrices must have one column per network output.');
    end
    if numel(lower_p) ~= size(lower_M, 1) || numel(upper_p) ~= size(upper_M, 1)
        error('Output specification vectors must match their matrices.');
    end
    if size(lower_M, 1) ~= size(upper_M, 1)
        error('Lower and upper output specifications must have the same number of rows.');
    end

    for l = numel(network.weights):-1:1
        W = network.weights{l};
        b = network.biases{l};
        lb = layer_bounds{l};

        [lower_M, lower_p, upper_M, upper_p] = backward_one_layer( ...
            lower_M, lower_p, upper_M, upper_p, W, b, ...
            lb.alpha_lower, lb.beta_lower, lb.alpha_upper, lb.beta_upper);
    end

    final_affine.lower_A = lower_M;
    final_affine.lower_c = lower_p;
    final_affine.upper_A = upper_M;
    final_affine.upper_c = upper_p;

    final_lower = affine_min(final_affine.lower_A, final_affine.lower_c, x0, eps_value);
    final_upper = affine_max(final_affine.upper_A, final_affine.upper_c, x0, eps_value);
end

function [new_lower_M, new_lower_p, new_upper_M, new_upper_p] = ...
    backward_one_layer(lower_M, lower_p, upper_M, upper_p, W, b, ...
                       alpha_l, beta_l, alpha_u, beta_u)
% Convert bounds over f^(l) into bounds over f^(l-1).
%
% Given
%     lower_M f^(l) + lower_p <= y <= upper_M f^(l) + upper_p,
% and linear activation bounds over f^(l), this implements conditional
% multiplication with positive/negative coefficient matrices.

    lower_M_pos = positive_part(lower_M);
    lower_M_neg = negative_part(lower_M);
    upper_M_pos = positive_part(upper_M);
    upper_M_neg = negative_part(upper_M);

    % Lower side: positive coefficients use activation lower bounds;
    % negative coefficients use activation upper bounds.
    lower_s_coeff = lower_M_pos .* alpha_l' + lower_M_neg .* alpha_u';
    new_lower_M = lower_s_coeff * W;
    new_lower_p = lower_M_pos * (alpha_l .* b + beta_l) + ...
                  lower_M_neg * (alpha_u .* b + beta_u) + lower_p;

    % Upper side: positive coefficients use activation upper bounds;
    % negative coefficients use activation lower bounds.
    upper_s_coeff = upper_M_pos .* alpha_u' + upper_M_neg .* alpha_l';
    new_upper_M = upper_s_coeff * W;
    new_upper_p = upper_M_pos * (alpha_u .* b + beta_u) + ...
                  upper_M_neg * (alpha_l .* b + beta_l) + upper_p;
end

%% ------------------------------------------------------------------------
% Activation relaxations
% -------------------------------------------------------------------------

function [alpha_l, beta_l, alpha_u, beta_u] = activation_relax(act, interval_lower, interval_upper)
    act_name = lower(act);
    switch act_name
        case 'relu'
            [alpha_l, beta_l, alpha_u, beta_u] = relu_relax(interval_lower, interval_upper);
        case 'sigmoid'
            [alpha_l, beta_l, alpha_u, beta_u] = sigmoid_relax(interval_lower, interval_upper);
        case 'linear'
            alpha_l = ones(size(interval_lower));
            beta_l = zeros(size(interval_lower));
            alpha_u = ones(size(interval_lower));
            beta_u = zeros(size(interval_lower));
        otherwise
            error('No relaxation registered for activation: %s', act);
    end
end

function [alpha_l, beta_l, alpha_u, beta_u] = relu_relax(lower, upper)
% Linear relaxation for ReLU over intervals [lower, upper].

    l = lower(:);
    u = upper(:);
    if any(l > u)
        error('Invalid interval: lower must be <= upper.');
    end

    alpha_l = zeros(size(l));
    beta_l = zeros(size(l));
    alpha_u = zeros(size(l));
    beta_u = zeros(size(l));

    positive = l >= 0;
    negative = u <= 0;
    crossing = ~(positive | negative);

    % Fully active: ReLU(s) = s.
    alpha_l(positive) = 1;
    alpha_u(positive) = 1;

    % Fully inactive: ReLU(s) = 0. Already initialized to zero.
    idx = crossing;
    denom = u(idx) - l(idx);
    alpha_u(idx) = u(idx) ./ denom;
    beta_u(idx) = -u(idx) .* l(idx) ./ denom;

    % CROWN/DeepPoly-style lower-bound choice.
    use_identity_lower = abs(l(idx)) < abs(u(idx));
    tmp = zeros(sum(idx), 1);
    tmp(use_identity_lower) = 1;
    alpha_l(idx) = tmp;
end

function [alpha_l, beta_l, alpha_u, beta_u] = sigmoid_relax(lower, upper)
% Linear relaxation for Sigmoid over intervals [lower, upper].
%
% Sigmoid is convex on (-inf, 0] and concave on [0, inf).
% For same-sign intervals, this uses the secant/tangent construction.
% For crossing intervals, tangent points are found by bisection.

    l = lower(:);
    u = upper(:);
    if any(l > u)
        error('Invalid interval: lower must be <= upper.');
    end

    n = numel(l);
    alpha_l = zeros(n, 1);
    beta_l = zeros(n, 1);
    alpha_u = zeros(n, 1);
    beta_u = zeros(n, 1);

    for i = 1:n
        li = l(i);
        ui = u(i);

        if abs(ui - li) < 1e-14
            slope = sigmoid_prime(li);
            intercept = sigmoid_fn(li) - slope * li;
            alpha_l(i) = slope;
            beta_l(i) = intercept;
            alpha_u(i) = slope;
            beta_u(i) = intercept;

        elseif li >= 0
            % Concave region: secant is lower, tangent is upper.
            slope_sec = (sigmoid_fn(ui) - sigmoid_fn(li)) / (ui - li);
            alpha_l(i) = slope_sec;
            beta_l(i) = sigmoid_fn(ui) - slope_sec * ui;

            xmid = 0.5 * (li + ui);
            slope_tan = sigmoid_prime(xmid);
            alpha_u(i) = slope_tan;
            beta_u(i) = sigmoid_fn(xmid) - slope_tan * xmid;

        elseif ui <= 0
            % Convex region: tangent is lower, secant is upper.
            xmid = 0.5 * (li + ui);
            slope_tan = sigmoid_prime(xmid);
            alpha_l(i) = slope_tan;
            beta_l(i) = sigmoid_fn(xmid) - slope_tan * xmid;

            slope_sec = (sigmoid_fn(ui) - sigmoid_fn(li)) / (ui - li);
            alpha_u(i) = slope_sec;
            beta_u(i) = sigmoid_fn(ui) - slope_sec * ui;

        else
            % Crossing interval.
            du = crossing_lower_tangent_point(li, ui);
            dl = crossing_upper_tangent_point(li, ui);

            slope_lower = sigmoid_prime(du);
            alpha_l(i) = slope_lower;
            beta_l(i) = sigmoid_fn(du) - slope_lower * du;

            slope_upper = sigmoid_prime(dl);
            alpha_u(i) = slope_upper;
            beta_u(i) = sigmoid_fn(dl) - slope_upper * dl;
        end

        % Numerical safety repair by dense sampling.
        xs = linspace(li, ui, 1001);
        ys = sigmoid_fn(xs);

        lower_line = alpha_l(i) * xs + beta_l(i);
        lower_violation = max(lower_line - ys);
        if lower_violation > 1e-10
            beta_l(i) = beta_l(i) - lower_violation - 1e-10;
        end

        upper_line = alpha_u(i) * xs + beta_u(i);
        upper_violation = max(ys - upper_line);
        if upper_violation > 1e-10
            beta_u(i) = beta_u(i) + upper_violation + 1e-10;
        end
    end
end

function du = crossing_lower_tangent_point(l, u)
% Find d_u in [l, 0] satisfying
%   (sigmoid(u) - sigmoid(d_u)) / (u - d_u) = sigmoid'(d_u).
    su = sigmoid_fn(u);
    fn = @(d) (su - sigmoid_fn(d)) ./ (u - d) - sigmoid_prime(d);
    du = bisect_root(fn, l, 0.0);
end

function dl = crossing_upper_tangent_point(l, u)
% Find d_l in [0, u] satisfying
%   (sigmoid(d_l) - sigmoid(l)) / (d_l - l) = sigmoid'(d_l).
    sl = sigmoid_fn(l);
    fn = @(d) (sigmoid_fn(d) - sl) ./ (d - l) - sigmoid_prime(d);
    dl = bisect_root(fn, 0.0, u);
end

function root = bisect_root(fn, lo, hi)
    tol = 1e-12;
    max_iter = 80;

    flo = fn(lo);
    fhi = fn(hi);

    if abs(flo) < tol
        root = lo;
        return;
    end
    if abs(fhi) < tol
        root = hi;
        return;
    end

    if flo * fhi > 0
        xs = linspace(lo, hi, 257);
        vals = arrayfun(fn, xs);
        [~, best] = min(abs(vals));

        bracket_found = false;
        for i = 1:(numel(xs)-1)
            if vals(i) == 0 || vals(i) * vals(i+1) <= 0
                lo = xs(i);
                hi = xs(i+1);
                flo = vals(i);
                fhi = vals(i+1); %#ok<NASGU>
                bracket_found = true;
                break;
            end
        end

        if ~bracket_found
            root = xs(best);
            return;
        end
    end

    for k = 1:max_iter %#ok<NASGU>
        mid = 0.5 * (lo + hi);
        fmid = fn(mid);
        if abs(fmid) < tol || abs(hi - lo) < tol
            root = mid;
            return;
        end
        if flo * fmid <= 0
            hi = mid;
        else
            lo = mid;
            flo = fmid;
        end
    end

    root = 0.5 * (lo + hi);
end

%% ------------------------------------------------------------------------
% Numeric helper functions
% -------------------------------------------------------------------------

function y = relu_fn(x)
    y = max(x, 0);
end

function y = sigmoid_fn(x)
% Numerically stable sigmoid for scalars, vectors, and matrices.
    y = zeros(size(x));
    pos = x >= 0;
    y(pos) = 1 ./ (1 + exp(-x(pos)));
    exp_x = exp(x(~pos));
    y(~pos) = exp_x ./ (1 + exp_x);
end

function y = sigmoid_prime(x)
    s = sigmoid_fn(x);
    y = s .* (1 - s);
end

function y = positive_part(x)
    y = max(x, 0);
end

function y = negative_part(x)
    y = min(x, 0);
end

function y = affine_min(A, c, x0, eps_value)
% Minimize A*x + c over x in [x0 - eps, x0 + eps].
    x0 = x0(:);
    eps_vec = expand_eps(eps_value, numel(x0));
    x_l = x0 - eps_vec;
    x_u = x0 + eps_vec;
    y = positive_part(A) * x_l + negative_part(A) * x_u + c;
end

function y = affine_max(A, c, x0, eps_value)
% Maximize A*x + c over x in [x0 - eps, x0 + eps].
    x0 = x0(:);
    eps_vec = expand_eps(eps_value, numel(x0));
    x_l = x0 - eps_vec;
    x_u = x0 + eps_vec;
    y = positive_part(A) * x_u + negative_part(A) * x_l + c;
end

function eps_vec = expand_eps(eps_value, n)
    if isscalar(eps_value)
        eps_vec = eps_value * ones(n, 1);
    else
        eps_vec = eps_value(:);
        if numel(eps_vec) ~= n
            error('eps vector dimension does not match x0.');
        end
    end
end

%% ------------------------------------------------------------------------
% Sanity tests
% -------------------------------------------------------------------------

function self_test_relaxations()
% Basic sampled checks that activation relaxations are sound.
    rng(0);

    acts = {'relu', 'sigmoid'};
    for a = 1:numel(acts)
        act = acts{a};
        for t = 1:200 %#ok<NASGU>
            vals = sort(-5 + 10 * rand(2, 1));
            lo = vals(1);
            hi = vals(2);
            if abs(hi - lo) < 1e-8
                hi = lo + 1e-6;
            end

            [alpha_l, beta_l, alpha_u, beta_u] = activation_relax(act, lo, hi);
            xs = linspace(lo, hi, 201);

            switch act
                case 'relu'
                    ys = relu_fn(xs);
                case 'sigmoid'
                    ys = sigmoid_fn(xs);
            end

            lhs = alpha_l(1) * xs + beta_l(1);
            rhs = alpha_u(1) * xs + beta_u(1);

            if any(lhs > ys + 1e-8)
                error('%s lower relaxation failed on interval [%.6f, %.6f].', act, lo, hi);
            end
            if any(ys > rhs + 1e-8)
                error('%s upper relaxation failed on interval [%.6f, %.6f].', act, lo, hi);
            end
        end
    end
end
