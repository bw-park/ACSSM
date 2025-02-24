import os
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s


def adjust_obs_for_extrapolation(obs, obs_valid, obs_times=None, cut_time=None):
    obs_valid_extrap = obs_valid.clone()
    obs_extrap = obs.clone()

    # zero out last half of observation (used for USHCN)
    if cut_time is None:
        n_observed_time_points = obs.shape[1] // 2
        obs_valid_extrap[:, n_observed_time_points:, ...] = False
        obs_extrap[:, n_observed_time_points:, ...] = 0

    # zero out observations at > cut_time (used for Physionet)
    else:
        mask_before_cut_time = obs_times < cut_time
        obs_valid_extrap *= mask_before_cut_time
        obs_extrap = torch.where(obs_valid_extrap[:, :, None].bool(), obs_extrap, 0.)

    return obs_extrap, obs_valid_extrap