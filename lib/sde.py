import numpy as np
import torch
import torch.nn as  nn
import geotorch
from typing import Tuple, Optional, Literal
from model import Transformer_Encoder, Decoder
from lib.jax_compat import associative_scan
from lib.losses import GNLL_

def elup(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j
    # return A_j * A_i, A_j * Bu_i + Bu_j
    return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_orthogonal(m):
    if type(m) == nn.Linear:
        nn.init.orthogonal_(m.weight, 1)        

class LinearSDE(torch.nn.Module):
    def __init__(self, args):
        super(LinearSDE, self).__init__()
        
        self.data = args.dataset
        self.task = args.task
        self.lamda_1 = args.lamda_1
        self.lamda_2 = args.lamda_2
        self.init_sigma = args.init_sigma
        self.ts = args.ts
        self.ld = args.state_dim
        self.nb = args.num_basis
        self.od = args.out_dim
        ### Initial
        
        #### Consturct base matrix for A.
        self.E = nn.Linear(self.ld, self.ld, bias=False)
        self.E.apply(init_orthogonal)
        geotorch.orthogonal(self.E, "weight")
        
        self.D = nn.Parameter(torch.randn(self.nb, self.ld))
        
        ### Init mean and covariance
        self.init_mean = torch.nn.Parameter(torch.randn(self.ld))
        self.init_log_var = torch.nn.Parameter(torch.randn(self.ld))
        self.y_log_var = torch.nn.Parameter(torch.randn(self.ld))
        
        #### Consturct coefficient net for A.
        self.coeff_net = nn.Sequential(nn.Linear(self.ld, self.nb),
                                       nn.Softmax(dim=-1))
        
        ### History encoder
        self.encoder = Transformer_Encoder(args)
        self.decoder = Decoder(args)
        

        self.B = nn.Linear(self.ld, self.ld, bias=False)
        self.B.apply(init_normal)
        self.C = nn.Linear(self.ld, self.ld, bias=False)
        self.C.apply(init_normal)
        self.M = nn.Linear(self.ld, self.ld, bias=False)
        self.M.apply(init_normal)

    def get_matrix(self, alpha, obs_times, sigma=1):
        
        Identity = torch.ones(alpha.shape[-1], device=alpha.device)
        
        A_basis = - (elup(self.D) + 1e-6)
        A_coeff = self.coeff_net(alpha)
        A_mat = (A_coeff[..., None] * A_basis[None]).sum(1)
        
        exp_A_mat_m = torch.exp(A_mat * obs_times)
        exp_B_mat_m = (1/A_mat) * (exp_A_mat_m - Identity) * alpha

        exp_A_mat_v = torch.exp(2 * A_mat * obs_times)
        exp_B_mat_v = 0.5 * sigma**2 * (1/A_mat) * (exp_A_mat_v - Identity) + 1e-6

        return torch.cat([exp_A_mat_m, exp_A_mat_v], dim=-1), torch.cat([exp_B_mat_m, exp_B_mat_v], dim=-1)

    def parallel_compute(self, init, E, Z, obs_times):
        
        alphas = torch.vmap(lambda u: self.B(u))(Z) # We assume that the control has already pre-computed  [ alphas := E.t() @ alphas ]

        mats_A, mats_B = torch.vmap(lambda a, t: self.get_matrix(a, t))(alphas, obs_times)
        
        cum_initial, cum_integral = associative_scan(binary_operator, (mats_A, mats_B))
        
        init_mean_var = torch.vmap(lambda cum_init : cum_init * init)(cum_initial)
        init_mean, init_var = torch.vmap(lambda mean_var : torch.chunk(mean_var, chunks=2, dim=1))(init_mean_var)
        xs_mean, xs_var = torch.vmap(lambda mean_var : torch.chunk(mean_var, chunks=2, dim=1))(cum_integral)
        
        y_var = (elup(self.y_log_var) + 1e-6)
        
        history = torch.vmap(lambda u: self.C(torch.vmap(lambda X: E(X))(u)))(Z) 
        means = torch.vmap(lambda mean : self.M(torch.vmap(lambda X: E(X))(mean)))(xs_mean + init_mean) + history # Adding history here enhanced the performance
        stds = torch.vmap(lambda std : self.M(torch.vmap(lambda X: E(X))(std)))(torch.sqrt(xs_var + init_var + y_var))

        return means, stds, alphas
    
    def forward(self, obs, obs_times, obs_valid, mask_obs, n_samples=3, epoch=None):
        # For training, we use ELBO in Appendix A.7
        obs_times_ = self.ts * obs_times
        Z, y_observed = self.encoder(obs, obs_times_, obs_mask=mask_obs,  event_mask=obs_valid) # Assume q_phi as dirac delta (do not sampling the latent obs.)
        
        obs_times = obs_times_[..., None]

        E = self.E.weight.data
        init_mean = E.t() @ self.init_mean
        init_var = E.t() @ (self.init_sigma * (elup(self.init_log_var) + 1e-6).diag_embed()) @ E
        init_var = torch.diag(init_var)
        
        init_mean_var = torch.cat([init_mean, init_var], dim=0) 
        
        means, stds, alphas = self.parallel_compute(init_mean_var, self.E, Z, obs_times)
        
        Z = torch.randn(size=(n_samples, *means.size())).to(means.device)
        Y = means + stds * Z

        observed_stamps = obs_valid[..., None]
        
        neg_log_potential = GNLL_(y_observed * observed_stamps, means * observed_stamps, stds**2, eps=1.)
        KLs =  0.5 * (alphas[:, :-1].pow(2) * (obs_times[:, 1:] - obs_times[:, :-1])).mean(0).sum() 
        L_alpha = self.lamda_1 * KLs + self.lamda_2 * neg_log_potential

        mean_out = self.decoder(Y) # Instead of directly decoding y_observed, we decode the samples sampled from controlled latent states.
        var_out = 0.01 * torch.ones_like(mean_out)
        
        return (mean_out, var_out), L_alpha
    