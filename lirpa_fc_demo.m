%% lirpa_fc_demo.m
% Forward and backward LiRPA implementation for fully connected networks.
%
% Assumptions:
%   1. Network layers are fully connected: s = W*f + b.
%   2. Hidden activations are ReLU.
%   3. The last layer may use Sigmoid.
%   4. Input perturbation is an L_infinity ball:
%          x_i in [x0_i - epsilon, x0_i + epsilon]
%
% This file is written as a self-contained MATLAB script with local functions.
% Run:
%   >> lirpa_fc_demo
%
% The activation-bound part is modularized through:
%   activationValue
%   activationInterval
%   activationLinearRelaxation
%
% To add another activation later, add a new case to these functions.

clear; clc;

net = makeXorNetwork();

tests = {
    [0;0], 0.100, 0, true;
    [0;0], 0.275, 0, false;
    [0;1], 0.100, 1, true;
    [1;0], 0.100, 1, true;
    [1;1], 0.100, 0, true;
};

fprintf('Forward LiRPA tests\n');
for k = 1:size(tests, 1)
    x0 = tests{k,1};
    eps = tests{k,2};
    label = tests{k,3};
    expectedRobust = tests{k,4};

    [ok, outInterval, trace] = verifyForwardBinary(net, x0, eps, label);

    fprintf('x=(%.1f,%.1f), eps=%.3f, label=%d, ', x0(1), x0(2), eps, label);
    fprintf('output=[%.6f, %.6f], robust=%d\n', ...
        outInterval.lower(1), outInterval.upper(1), ok);

    assert(ok == expectedRobust);
end

fprintf('\nBackward LiRPA tests\n');
for k = 1:size(tests, 1)
    x0 = tests{k,1};
    eps = tests{k,2};
    label = tests{k,3};
    expectedRobust = tests{k,4};

    [ok, outInterval, logitInterval, affine, trace] = ...
        verifyBackwardBinary(net, x0, eps, label);

    fprintf('x=(%.1f,%.1f), eps=%.3f, label=%d, ', x0(1), x0(2), eps, label);
    fprintf('logit=[%.6f, %.6f], output=[%.6f, %.6f], robust=%d\n', ...
        logitInterval.lower(1), logitInterval.upper(1), ...
        outInterval.lower(1), outInterval.upper(1), ok);

    assert(ok == expectedRobust);
end

fprintf('\nDetailed backward trace for x=(0,0), eps=0.275\n');
x0 = [0;0];
eps = 0.275;
[ok, outInterval, logitInterval, affine, trace] = verifyBackwardBinary(net, x0, eps, 0);

fprintf('logit interval  = [%.9f, %.9f]\n', ...
    logitInterval.lower(1), logitInterval.upper(1));
fprintf('output interval = [%.9f, %.9f]\n', ...
    outInterval.lower(1), outInterval.upper(1));
fprintf('robust          = %d\n', ok);

fprintf('symbolic lower bound:\n');
fprintf('  [%.9f %.9f] * x + %.9f\n', ...
    affine.lowerCoeff(1), affine.lowerCoeff(2), affine.lowerBias);
fprintf('symbolic upper bound:\n');
fprintf('  [%.9f %.9f] * x + %.9f\n', ...
    affine.upperCoeff(1), affine.upperCoeff(2), affine.upperBias);

fprintf('\nAll tests passed.\n');


%% Network construction

function net = makeXorNetwork()
% XOR network from the LiRPA note:
%
% s1 = [ 2.1247   2.1267  ] x + [-2.1259]
%      [-2.1237  -2.1235  ]     [ 2.1234]
% f1 = ReLU(s1)
%
% s2 = [-3.6788  -3.6766] f1 + 3.5451
% f2 = sigmoid(s2)

    layer1.W = [ 2.1247,  2.1267;
                -2.1237, -2.1235 ];
    layer1.b = [-2.1259; 2.1234];
    layer1.activation = 'relu';

    layer2.W = [-3.6788, -3.6766];
    layer2.b = 3.5451;
    layer2.activation = 'sigmoid';

    net.layers = {layer1, layer2};
end


%% Forward execution

function y = networkForward(net, x)
    h = x;
    for i = 1:numel(net.layers)
        layer = net.layers{i};
        s = layer.W * h + layer.b;
        h = activationValue(layer.activation, s);
    end
    y = h;
end


%% Forward LiRPA

function [outInterval, trace] = forwardLiRPA(net, x0, eps)
% Numeric forward bound propagation.
%
% Input:
%   x0  : nominal input column vector
%   eps : L_infinity perturbation radius
%
% Output:
%   outInterval.lower <= network(x) <= outInterval.upper
%   for all x in [x0-eps, x0+eps].

    if eps < 0
        error('epsilon must be nonnegative');
    end

    current.lower = x0 - eps;
    current.upper = x0 + eps;

    trace = cell(numel(net.layers), 1);

    for i = 1:numel(net.layers)
        layer = net.layers{i};

        pre = affineInterval(layer.W, layer.b, current);
        post = activationInterval(layer.activation, pre);

        trace{i}.pre = pre;
        trace{i}.post = post;
        trace{i}.activation = layer.activation;

        current = post;
    end

    outInterval = current;
end

function interval = affineInterval(W, b, inputInterval)
% Bound s = W*x + b.
%
% Let Wpos = max(W,0), Wneg = min(W,0).
% Then:
%   lower = Wpos*L + Wneg*U + b
%   upper = Wpos*U + Wneg*L + b

    Wpos = max(W, 0);
    Wneg = min(W, 0);

    interval.lower = Wpos * inputInterval.lower + Wneg * inputInterval.upper + b;
    interval.upper = Wpos * inputInterval.upper + Wneg * inputInterval.lower + b;
end

function [ok, outInterval, trace] = verifyForwardBinary(net, x0, eps, expectedLabel)
% Verify one-output sigmoid binary classification using output bounds.
%
% expectedLabel = 0:
%   robust if output upper < 0.5
%
% expectedLabel = 1:
%   robust if output lower >= 0.5

    [outInterval, trace] = forwardLiRPA(net, x0, eps);

    if numel(outInterval.lower) ~= 1
        error('Binary verification expects a scalar output.');
    end

    if expectedLabel == 0
        ok = outInterval.upper(1) < 0.5;
    elseif expectedLabel == 1
        ok = outInterval.lower(1) >= 0.5;
    else
        error('expectedLabel must be 0 or 1.');
    end
end


%% Backward LiRPA

function [logitInterval, affine, trace] = backwardLiRPALogit(net, x0, eps, outputIndex)
% Symbolic backward LiRPA for the final pre-sigmoid logit.
%
% For a binary network ending with:
%     z = W_last*h + b_last
%     y = sigmoid(z)
%
% this function computes affine lower/upper bounds:
%     lowerCoeff' * x + lowerBias <= z(x)
%     z(x) <= upperCoeff' * x + upperBias
%
% and then evaluates them over x in [x0-eps, x0+eps].

    if nargin < 4
        outputIndex = 1;
    end

    if eps < 0
        error('epsilon must be nonnegative');
    end

    inputInterval.lower = x0 - eps;
    inputInterval.upper = x0 + eps;

    % Forward pass is used to obtain pre-activation intervals for
    % activation relaxations.
    [~, trace] = forwardLiRPA(net, x0, eps);

    nLayers = numel(net.layers);
    lastLayer = net.layers{nLayers};

    if ~strcmp(lastLayer.activation, 'sigmoid')
        error('backwardLiRPALogit expects the last activation to be sigmoid.');
    end

    if outputIndex < 1 || outputIndex > size(lastLayer.W, 1)
        error('Invalid outputIndex.');
    end

    % Start from the selected pre-sigmoid logit:
    % z_j = W_last(j,:)*f + b_last(j).
    lowerCoeff = lastLayer.W(outputIndex, :)';
    upperCoeff = lastLayer.W(outputIndex, :)';
    lowerBias = lastLayer.b(outputIndex);
    upperBias = lastLayer.b(outputIndex);

    % Propagate backward through hidden layers.
    for layerIndex = nLayers-1:-1:1
        layer = net.layers{layerIndex};
        preInterval = trace{layerIndex}.pre;

        relax = activationLinearRelaxation(layer.activation, preInterval);

        [lowerCoeff, lowerBias] = backwardOneActivation( ...
            lowerCoeff, lowerBias, relax, 'lower');
        [lowerCoeff, lowerBias] = backwardOneDense( ...
            lowerCoeff, lowerBias, layer);

        [upperCoeff, upperBias] = backwardOneActivation( ...
            upperCoeff, upperBias, relax, 'upper');
        [upperCoeff, upperBias] = backwardOneDense( ...
            upperCoeff, upperBias, layer);
    end

    logitLower = evaluateAffineLower(lowerCoeff, lowerBias, inputInterval);
    logitUpper = evaluateAffineUpper(upperCoeff, upperBias, inputInterval);

    logitInterval.lower = logitLower;
    logitInterval.upper = logitUpper;

    affine.lowerCoeff = lowerCoeff;
    affine.lowerBias = lowerBias;
    affine.upperCoeff = upperCoeff;
    affine.upperBias = upperBias;
end

function [newCoeff, newBias] = backwardOneActivation(coeff, bias, relax, mode)
% Backpropagate affine expression through an activation relaxation.
%
% Suppose:
%   expr = coeff' * f + bias
%   f_i = activation(s_i)
%
% For a lower bound:
%   coeff_i >= 0 uses the lower relaxation of f_i.
%   coeff_i <  0 uses the upper relaxation of f_i.
%
% For an upper bound:
%   coeff_i >= 0 uses the upper relaxation of f_i.
%   coeff_i <  0 uses the lower relaxation of f_i.

    cpos = max(coeff, 0);
    cneg = min(coeff, 0);

    switch mode
        case 'lower'
            newCoeff = cpos .* relax.lowerSlope + cneg .* relax.upperSlope;
            newBias = bias ...
                + cpos' * relax.lowerBias ...
                + cneg' * relax.upperBias;

        case 'upper'
            newCoeff = cpos .* relax.upperSlope + cneg .* relax.lowerSlope;
            newBias = bias ...
                + cpos' * relax.upperBias ...
                + cneg' * relax.lowerBias;

        otherwise
            error('mode must be lower or upper.');
    end
end

function [newCoeff, newBias] = backwardOneDense(coeffOnPre, bias, layer)
% Substitute s = W*h + b into coeffOnPre' * s + bias.
%
% coeffOnPre' * (W*h + b) + bias
% = (W' * coeffOnPre)' * h + coeffOnPre' * b + bias

    newCoeff = layer.W' * coeffOnPre;
    newBias = bias + coeffOnPre' * layer.b;
end

function val = evaluateAffineLower(coeff, bias, inputInterval)
% min_x coeff' * x + bias over x in [L,U].

    cpos = max(coeff, 0);
    cneg = min(coeff, 0);

    val = cpos' * inputInterval.lower + cneg' * inputInterval.upper + bias;
end

function val = evaluateAffineUpper(coeff, bias, inputInterval)
% max_x coeff' * x + bias over x in [L,U].

    cpos = max(coeff, 0);
    cneg = min(coeff, 0);

    val = cpos' * inputInterval.upper + cneg' * inputInterval.lower + bias;
end

function [outInterval, logitInterval, affine, trace] = backwardLiRPAOutput(net, x0, eps, outputIndex)
% Bound both the pre-sigmoid logit and the final sigmoid output.

    if nargin < 4
        outputIndex = 1;
    end

    [logitInterval, affine, trace] = backwardLiRPALogit(net, x0, eps, outputIndex);

    outInterval = activationInterval('sigmoid', logitInterval);
end

function [ok, outInterval, logitInterval, affine, trace] = verifyBackwardBinary(net, x0, eps, expectedLabel)
% Verify binary classification using the pre-sigmoid logit.
%
% Because sigmoid(z) >= 0.5 iff z >= 0:
%
% expectedLabel = 0:
%   robust if logit upper < 0
%
% expectedLabel = 1:
%   robust if logit lower >= 0

    [outInterval, logitInterval, affine, trace] = backwardLiRPAOutput(net, x0, eps, 1);

    if expectedLabel == 0
        ok = logitInterval.upper(1) < 0;
    elseif expectedLabel == 1
        ok = logitInterval.lower(1) >= 0;
    else
        error('expectedLabel must be 0 or 1.');
    end
end


%% Activation abstraction module

function y = activationValue(name, x)
    switch lower(name)
        case 'identity'
            y = x;

        case 'relu'
            y = max(0, x);

        case 'sigmoid'
            y = sigmoid(x);

        otherwise
            error('Unknown activation: %s', name);
    end
end

function interval = activationInterval(name, preInterval)
% Concrete output interval of activation over preInterval.
%
% This function is exact for ReLU and Sigmoid.
% For future activations, add a new case here.

    switch lower(name)
        case 'identity'
            interval = preInterval;

        case 'relu'
            interval.lower = max(0, preInterval.lower);
            interval.upper = max(0, preInterval.upper);

        case 'sigmoid'
            % Sigmoid is monotone increasing.
            interval.lower = sigmoid(preInterval.lower);
            interval.upper = sigmoid(preInterval.upper);

        otherwise
            error('Unknown activation: %s', name);
    end
end

function relax = activationLinearRelaxation(name, preInterval)
% Linear lower/upper relaxation of activation over [l,u].
%
% The returned fields satisfy:
%   lowerSlope .* s + lowerBias <= activation(s)
%   activation(s) <= upperSlope .* s + upperBias
%
% All fields are column vectors.

    switch lower(name)
        case 'identity'
            z = zeros(size(preInterval.lower));
            o = ones(size(preInterval.lower));

            relax.lowerSlope = o;
            relax.lowerBias = z;
            relax.upperSlope = o;
            relax.upperBias = z;

        case 'relu'
            relax = reluLinearRelaxation(preInterval);

        case 'sigmoid'
            relax = sigmoidLinearRelaxationBasic(preInterval);

        otherwise
            error('Unknown activation: %s', name);
    end
end

function relax = reluLinearRelaxation(preInterval)
% ReLU relaxation.
%
% Cases:
%   1. 0 <= l <= u:
%        lower = s, upper = s
%   2. l <= u <= 0:
%        lower = 0, upper = 0
%   3. l <= 0 <= u and |l| <= |u|:
%        lower = 0
%        upper = (u/(u-l))*s - (u*l)/(u-l)
%   4. l <= 0 <= u and |l| > |u|:
%        lower = s
%        upper = (u/(u-l))*s - (u*l)/(u-l)

    l = preInterval.lower;
    u = preInterval.upper;

    relax.lowerSlope = zeros(size(l));
    relax.lowerBias = zeros(size(l));
    relax.upperSlope = zeros(size(l));
    relax.upperBias = zeros(size(l));

    pos = l >= 0;
    neg = u <= 0;
    cross = ~(pos | neg);

    % Fully positive: ReLU(s) = s.
    relax.lowerSlope(pos) = 1;
    relax.upperSlope(pos) = 1;

    % Fully negative: already zero.

    % Crossing zero.
    denom = u(cross) - l(cross);
    relax.upperSlope(cross) = u(cross) ./ denom;
    relax.upperBias(cross) = -u(cross) .* l(cross) ./ denom;

    % Lower-bound heuristic:
    % ReLU(s) >= 0 and ReLU(s) >= s are both valid.
    % Use s when |l| > |u|, otherwise use 0.
    useIdentityLower = abs(l(cross)) > abs(u(cross));

    tmp = zeros(size(denom));
    tmp(useIdentityLower) = 1;
    relax.lowerSlope(cross) = tmp;
end

function relax = sigmoidLinearRelaxationBasic(preInterval)
% Basic modular sigmoid relaxation.
%
% Forward interval propagation uses activationInterval('sigmoid', ...)
% and is exact because sigmoid is monotone.
%
% Backward binary verification in this demo bounds the pre-sigmoid logit,
% so this function is mainly a placeholder for future extensions.
%
% For a sharper CROWN-style sigmoid relaxation, replace the crossing-zero
% case with a binary-search tangent construction.

    l = preInterval.lower;
    u = preInterval.upper;
    mid = (l + u) / 2;

    sigL = sigmoid(l);
    sigU = sigmoid(u);
    width = max(u - l, 1e-12);

    secantSlope = (sigU - sigL) ./ width;
    secantBias = sigU - secantSlope .* u;

    tangentSlope = sigmoidDerivative(mid);
    tangentBias = sigmoid(mid) - tangentSlope .* mid;

    relax.lowerSlope = zeros(size(l));
    relax.lowerBias = zeros(size(l));
    relax.upperSlope = zeros(size(l));
    relax.upperBias = zeros(size(l));

    pos = l >= 0;
    neg = u <= 0;
    cross = ~(pos | neg);

    % On x <= 0, sigmoid is convex:
    % tangent lower, secant upper.
    relax.lowerSlope(neg) = tangentSlope(neg);
    relax.lowerBias(neg) = tangentBias(neg);
    relax.upperSlope(neg) = secantSlope(neg);
    relax.upperBias(neg) = secantBias(neg);

    % On x >= 0, sigmoid is concave:
    % secant lower, tangent upper.
    relax.lowerSlope(pos) = secantSlope(pos);
    relax.lowerBias(pos) = secantBias(pos);
    relax.upperSlope(pos) = tangentSlope(pos);
    relax.upperBias(pos) = tangentBias(pos);

    % Simple placeholder for crossing-zero intervals.
    relax.lowerSlope(cross) = secantSlope(cross);
    relax.lowerBias(cross) = secantBias(cross);
    relax.upperSlope(cross) = secantSlope(cross);
    relax.upperBias(cross) = secantBias(cross);
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end

function y = sigmoidDerivative(x)
    s = sigmoid(x);
    y = s .* (1 - s);
end
