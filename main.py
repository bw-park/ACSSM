import configs
import torch

import random
import os
import numpy as np
import argparse
import wandb

from lib.data_utils import load_data
from lib.utils import set_seed, count_parameters
from lib.amortized_control_ssm import ACSSM

parser = argparse.ArgumentParser('ACSSM')
parser.add_argument('--epochs', type=int, default=None, help="Number of epochs.")
parser.add_argument('--lr',  type=float, default=None, help="Learning rate.")
parser.add_argument("--gpu", type=int, default=0, help="GPU device")
parser.add_argument('--problem_name', type=str, default=None, help="Problem to solve")
parser.add_argument('--info_type', type=str, default="full", help="Assimilation type to encode the information")
parser.add_argument('--cut-time', type=int, default=None, help='Timepoint at which extrapolation starts.')
parser.add_argument('-b', '--batch-size', type=int, default=None, help="Batch size for training and test set.")
parser.add_argument('--task', type=str, default=None, help="Target task.")
parser.add_argument('--dataset', type=str, default=None, help="Dataset to use. Available datasets are physionet, ushcn and pendulum.")
parser.add_argument('--sample-rate', type=float, default=None, help='Sample time points to increase irregularity of timestamps. For example, if sample_rate=0.5 half of the time points are discarded at random in the data preprocessing.')
parser.add_argument('--impute-rate', type=float, default=None, help='Remove time points for interpolation. For example, if impute_rate=0.3 the model is given 70% of the time points and tasked to reconstruct the entire series.')
parser.add_argument('--unobserved-rate', type=float, default=0.2, help='Percentage of features to remove per timestamp in order to increase sparseness across dimensions (applied only for USHCN)')
parser.add_argument('--data-random-seed', type=int, default=None, help="Random seed for subsampling timepoints and features.")
parser.add_argument('-rs', '--random-seed', type=int, default=42, help="Random seed for initializing model parameters.")
parser.add_argument('--num-workers', type=int, default=None, help="Number of workers to use in dataloader.")
parser.add_argument('--pin-memory', type=bool, default=True, help="If to pin memory in dataloader.")
parser.add_argument('--state-dim', type=int, default=None, help="Dimension of latent states")
parser.add_argument('--out-dim', type=int, default=None, help="Dimension of output")
parser.add_argument('--n_layer', type=int, default=None, help="Number of layer for transformer")
parser.add_argument('--drop_out', type=float, default=None, help="Dropout rate for transformer")
parser.add_argument('--lamda_1', type=float, default=None, help="Adjusting the lagrange term in ELBO")
parser.add_argument('--lamda_2', type=float, default=None, help="Adjusting the mayer term in ELBO")
parser.add_argument('--init_sigma', type=float, default=None, help="Adjusting initial covaraince of latent dynamics")
parser.add_argument('--ts', type=float, default=None, help="Time scaler")
parser.add_argument('--num-basis', type=int, default=None, help="Number of basis matrices to use in transition model for locally-linear transitions. L in paper")

problem_name = parser.parse_args().problem_name
default_config = {
    'pendulum_regression':     configs.get_pendulum_regression_configs,
    'physionet_interpolation': configs.get_physionet_interpolation_configs,
    'physionet_extrapolation': configs.get_physionet_extrapolation_configs,
    'ushcn_interpolation':     configs.get_ushcn_interpolation_configs,
    'ushcn_extrapolation':     configs.get_ushcn_extrapolation_configs,   
    'person_activity_classification': configs.get_person_activity_classification_configs,
    
}.get(problem_name)()
parser.set_defaults(**default_config)

args = parser.parse_args()

args.device = 'cuda:' + str(args.gpu)
seed = args.random_seed
set_seed(seed)


def main(args):
    # ========= print options =========
    for o in vars(args):
        print("#", o, ":", getattr(args, o))
    run = ACSSM(args)
    wandb.init(project="acssm", config=args, save_code=True, mode="online",
               name=f'{args.problem_name}')
    print((f"# param of model: {count_parameters(run.dynamics)}"))
    train_dl, valid_dl = load_data(args)
    run.train_and_eval(train_dl, valid_dl)
    
main(args)