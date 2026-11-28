function [H, split, info] = quadriga(cfg)
%QUADRIGA Generate normalized single-BS, single-snapshot user channels.
%
% Flat output:
%   H: [Nrx, Ntx, Nsc, Nuser]
% Factorized output:
%   H: [NrxX, NrxY, NtxX, NtxY, Nsc, Nuser]
%
% Absolute path delays are retained. Absolute link power is intentionally
% discarded. Each user channel is normalized to unit Frobenius norm, so
% subsequent noise can be specified through a relative SNR. If requested,
% info contains only the compact delay-window statistic needed to select the
% delay dictionary length; individual path parameters are not retained.

validateQuadrigaConfig(cfg);
%qdRoot = resolveQuadrigaRoot(cfg.quadrigaRoot);
%addpath(genpath(qdRoot));

oldRng = rng;
rngCleanup = onCleanup(@() rng(oldRng));
rng(cfg.seed, "twister");

simpar = qd_simulation_parameters;
simpar.center_frequency = cfg.centerFrequency;
simpar.show_progress_bars = logical(cfg.showProgress);
simpar.use_absolute_delays = 1;
simpar.use_3GPP_baseline = 0;

layout = qd_layout(simpar);
layout.name = 'single_bs_many_users';
layout.no_tx = 1;
layout.no_rx = cfg.numUsers;
layout.tx_position(:, 1) = [0; 0; cfg.bsHeight];

txArray = makeUpa(cfg.txSize, cfg.txSpacing, ...
    cfg.centerFrequency);
rxArray = makeUpa(cfg.rxSize, cfg.rxSpacing, ...
    cfg.centerFrequency);
layout.tx_array(1) = txArray;
layout.rx_array = rxArray;

layout.rx_position = makeUserPositions(cfg.distance, cfg.ueHeight, ...
    cfg.numUsers);
layout.set_scenario(char(cfg.scenario));
layout.set_pairing('all');

channels = layout.get_channels;
nRx = prod(cfg.rxSize);
nTx = prod(cfg.txSize);
nSc = cfg.numSubcarriers;
Hflat = complex(zeros(nRx, nTx, nSc, cfg.numUsers));
if nargout > 2
    info.delayEnergyFraction = cfg.delayEnergyFraction;
    info.delayTailNmseDb = 10*log10(1-info.delayEnergyFraction);
    info.delayWindowSeconds = zeros(1, cfg.numUsers);
    info.compactDelayWindowSeconds = zeros(1, cfg.numUsers);
    info.absoluteDelays = true;
end

bandwidth = nSc * cfg.subcarrierSpacing;
carrierGrid = ((0:nSc-1) - floor(nSc/2)) / nSc;
for userIdx = 1:cfg.numUsers
    if nargout > 2
        [info.delayWindowSeconds(userIdx), ...
            info.compactDelayWindowSeconds(userIdx)] = delayWindows( ...
            channels(userIdx, 1), info.delayEnergyFraction);
    end
    userChannel = channels(userIdx, 1).fr(bandwidth, carrierGrid, 1);
    userChannel = reshape(userChannel, nRx, nTx, nSc);
    channelNorm = norm(userChannel(:));
    if channelNorm == 0
        error("channelgen:quadriga:ZeroChannel", ...
            "QuaDRiGa returned a zero channel for user %d.", userIdx);
    end
    Hflat(:, :, :, userIdx) = userChannel / channelNorm;
end

switch lower(string(cfg.outputShape))
    case "flat"
        H = Hflat;
    case "factorized"
        H = reshape(Hflat, [cfg.rxSize, cfg.txSize, nSc, cfg.numUsers]);
    otherwise
        error("channelgen:quadriga:OutputShape", ...
            "outputShape must be 'flat' or 'factorized'.");
end

rng(cfg.seed + 1, "twister");
order = randperm(cfg.numUsers);
nTuning = round(cfg.tuningFraction * cfg.numUsers);
if cfg.tuningFraction > 0 && cfg.numUsers > 1
    nTuning = max(1, min(cfg.numUsers - 1, nTuning));
else
    nTuning = 0;
end
split.tuning = sort(order(1:nTuning));
split.evaluation = sort(order(nTuning+1:end));
end

function [absoluteLimit, compactWindow] = delayWindows( ...
        channel, energyFraction)
coeff = channel.coeff;
delay = channel.delay;

if channel.individual_delays
    delay = reshape(delay, [], 1);
    power = reshape(abs(coeff).^2, [], 1);
else
    delay = reshape(delay(:, 1), [], 1);
    power = reshape(sum(abs(coeff(:,:,:,1)).^2, [1 2]), [], 1);
end

[power, valid] = deal(power(:), isfinite(delay) & isfinite(power) & power > 0);
delay = delay(valid);
power = power(valid);
[delay, order] = sort(delay);
power = power(order);
totalPower = sum(power);
if totalPower == 0
    absoluteLimit = 0;
    compactWindow = 0;
    return
end
targetPower = energyFraction*totalPower;
idx = find(cumsum(power) >= targetPower, 1);
absoluteLimit = delay(idx);

compactWindow = Inf;
right = 1;
windowPower = 0;
for left = 1:numel(delay)
    while right <= numel(delay) && windowPower < targetPower
        windowPower = windowPower+power(right);
        right = right+1;
    end
    if windowPower >= targetPower
        compactWindow = min(compactWindow, delay(right-1)-delay(left));
    end
    windowPower = windowPower-power(left);
end
end

function array = makeUpa(arraySize, spacingLambda, centerFrequency)
c0 = 299792458;
lambda = c0 / centerFrequency;
x = ((0:arraySize(1)-1) - (arraySize(1)-1)/2) ...
    * spacingLambda(1) * lambda;
y = ((0:arraySize(2)-1) - (arraySize(2)-1)/2) ...
    * spacingLambda(2) * lambda;

positions = zeros(3, prod(arraySize));
for ix = 1:arraySize(1)
    for iy = 1:arraySize(2)
        elementIdx = sub2ind(arraySize, ix, iy);
        positions(2, elementIdx) = x(ix);
        positions(3, elementIdx) = y(iy);
    end
end

array = qd_arrayant('patch');
array.no_elements = prod(arraySize);
array.element_position = positions;
array.center_frequency = centerFrequency;
end

function positions = makeUserPositions(distance, ueHeight, numUsers)
if isscalar(distance)
    radius = repmat(distance, 1, numUsers);
else
    radius = sqrt(distance(1)^2 + rand(1, numUsers) ...
        * (distance(2)^2 - distance(1)^2));
end
azimuth = 2*pi*rand(1, numUsers);
positions = [radius .* cos(azimuth); ...
             radius .* sin(azimuth); ...
             repmat(ueHeight, 1, numUsers)];
end

function qdRoot = resolveQuadrigaRoot(configuredRoot)
if strlength(string(configuredRoot)) > 0
    qdRoot = char(configuredRoot);
else
    packageDir = fileparts(mfilename("fullpath"));
    repositoryDir = fileparts(fileparts(packageDir));
    qdRoot = fullfile(repositoryDir, "..", "QuaDRiGa", "quadriga_src");
end
if ~isfolder(qdRoot)
    error("channelgen:quadriga:MissingDependency", ...
        "QuaDRiGa was not found at '%s'. Set cfg.quadrigaRoot.", qdRoot);
end
end

function validateQuadrigaConfig(cfg)
required = ["scenario", "centerFrequency", "subcarrierSpacing", ...
    "numSubcarriers", "txSize", "rxSize", "txSpacing", ...
    "rxSpacing", "bsHeight", "ueHeight", "distance", ...
    "numUsers", "seed", "outputShape", "tuningFraction", ...
    "delayEnergyFraction", "showProgress", "quadrigaRoot"];
for field = required
    if ~isfield(cfg, field)
        error("channelgen:quadriga:MissingField", ...
            "Missing configuration field cfg.%s.", field);
    end
end
validateattributes(cfg.centerFrequency, {'numeric'}, {'scalar', 'positive'});
validateattributes(cfg.subcarrierSpacing, {'numeric'}, {'scalar', 'positive'});
validateattributes(cfg.numSubcarriers, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(cfg.numUsers, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});
validateattributes(cfg.seed, {'numeric'}, ...
    {'scalar', 'integer', 'nonnegative'});
validateattributes(cfg.delayEnergyFraction, {'numeric'}, ...
    {'scalar', '>', 0, '<', 1});
validateattributes(cfg.txSize, {'numeric'}, ...
    {'vector', 'numel', 2, 'integer', 'positive'});
validateattributes(cfg.rxSize, {'numeric'}, ...
    {'vector', 'numel', 2, 'integer', 'positive'});
validateattributes(cfg.txSpacing, {'numeric'}, ...
    {'vector', 'numel', 2, 'positive'});
validateattributes(cfg.rxSpacing, {'numeric'}, ...
    {'vector', 'numel', 2, 'positive'});
validateattributes(cfg.distance, {'numeric'}, {'vector', 'positive'});
if numel(cfg.distance) > 2 || ...
        (numel(cfg.distance) == 2 && cfg.distance(1) > cfg.distance(2))
    error("channelgen:quadriga:Distance", ...
        "distance must be a positive scalar or [minimum maximum].");
end
validateattributes(cfg.tuningFraction, {'numeric'}, ...
    {'scalar', '>=', 0, '<', 1});
if ~any(strcmpi(cfg.outputShape, ["flat", "factorized"]))
    error("channelgen:quadriga:OutputShape", ...
        "outputShape must be 'flat' or 'factorized'.");
end
end
