
function H = GenRealisticChannel(seed)
    addpath(genpath("C:\Users\denis\OneDrive\Desktop\MISC\Libs"))
    cfg = profile_quadriga("quadriga", "uma_nlos", seed);
    cfg.quadrigaRoot = "C:\Users\denis\OneDrive\Desktop\MISC\Libs\quadriga";
    [H, split, info] = quadriga(cfg);

    H = double(tenmat(H, 3));
    H = H/norm(H, 'fro')*sqrt(numel(H));
end