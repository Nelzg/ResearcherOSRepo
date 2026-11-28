
function H = GenRealisticChannel(seed)
    addpath(genpath("\Libs"))
    cfg = profile_quadriga("quadriga", "uma_nlos", seed);
    cfg.quadrigaRoot = "\Libs\quadriga";
    [H, split, info] = quadriga(cfg);

    H = double(tenmat(H, 3));
    H = H/norm(H, 'fro')*sqrt(numel(H));
end