clc
clear all
%
rng_control_value = 1;
addpath(genpath("C:\Users\denis\OneDrive\Desktop\MISC\Libs"))
cfg = profile_quadriga("quadriga", "uma_nlos", rng_control_value);
cfg.quadrigaRoot = "C:\Users\denis\OneDrive\Desktop\MISC\Libs\quadriga";
[H, split, info] = quadriga(cfg);
