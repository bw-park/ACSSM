import os
import time
import wandb
import torch
import torch.nn as nn

import lib.sde as sde
from lib.losses import MSE_, GNLL_, BNLL_, CNLL_
from lib.data_utils import adjust_obs_for_extrapolation
from lib.utils import get_time

class ACSSM():
    def __init__(self, args):
        super(ACSSM, self).__init__()
        self.n_epoch = args.epochs
        self.task = args.task
        self.cut_time = args.cut_time
        self.device = args.device
        self.dataset = args.dataset
        
        self.dynamics = sde.LinearSDE(args)
        self.dynamics = self.dynamics.to(self.device)
        self.optimizer = torch.optim.AdamW(self.dynamics.parameters(), lr=args.lr, weight_decay=args.wd)
        
    
    def train_and_eval(self, train_dl, eval_dl):
        
        self.start_time=time.time()
        for epoch in range(self.n_epoch):
            epoch_ll = 0
            epoch_mse = 0
            epoch_loss = 0
            num_data = 0
            for _, data in enumerate(train_dl):
                
                if self.dataset == 'pendulum':
                    if self.task == 'regression':
                        obs, truth, obs_times, obs_valid = [j.to(self.device).to(torch.float32) for j in data]
                        mask_obs = None
                        mask_truth = None
                    else:
                        obs, truth, obs_valid, obs_times, mask_truth = [j.to(self.device).to(torch.float32) for j in data]
                        mask_obs = None
                elif self.dataset == 'person_activity':
                    obs = data['inp_obs'].to(self.device)
                    truth = data['evd_obs'].to(self.device)
                    obs_times = data['inp_tid'].to(self.device)
                    labels = data['aux_obs'].to(self.device)
                    b, t, _ = obs.size()
                    obs_valid = torch.ones(b, t).to(obs.device)
                    mask_obs = None
                    mask_truth = None
                else:
                    obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [j.to(self.device).to(torch.float32) for j in data]
                    if self.task == 'extrapolation':
                        obs, obs_valid = adjust_obs_for_extrapolation(obs, obs_valid, obs_times, self.cut_time)

                self.optimizer.zero_grad()
                out, L_alpha = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=3, epoch=epoch)
                mean, var = out

                # Example loss
                batch_len = truth.size(0)

                if self.dataset == 'pendulum' and self.task == 'interpolation':
                    train_nll = BNLL_(truth, mean) * batch_len
                    train_mse = MSE_(truth.flatten(start_dim=2), mean.flatten(start_dim=3), mask = mask_truth.flatten(start_dim=2)) * batch_len
                elif self.task == 'classification':
                    train_nll, train_mse = CNLL_(labels, mean)
                else:
                    train_nll = GNLL_(truth, mean, var, mask = mask_truth) * batch_len
                    train_mse = MSE_(truth, mean, mask = mask_truth) * batch_len
                    
                loss = train_nll + L_alpha
                
                loss.backward()
                if self.task == 'classification':
                    nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1)
                self.optimizer.step()
                
                epoch_mse += train_mse.item()
                epoch_ll += train_nll.item()
                epoch_loss += loss.item()
                num_data += batch_len
                
            print('--------------[ {} || {} ]--------------'.format(epoch, self.n_epoch))
            print('[Time elapsed] : {0}:{1:02d}:{2:05.2f}'.format(*get_time(time.time() - self.start_time)))      
            with torch.no_grad():
                test_mse, test_nll, impute_mse, impute_nll = self.eval_func(eval_dl)
            
            if self.task == 'classification':
                print("[Train  ] NLL : {:.6f} || ACC : {:.6f}".format(epoch_ll/num_data, epoch_mse/num_data))
                print("[Eval  ] NLL : {:.6f} || ACC : {:.6f}".format(test_nll, test_mse))
                wandb.log({"train_acc" : (epoch_mse/num_data)}, step=epoch)
                wandb.log({"eval_acc" : test_mse}, step=epoch)

            else:
                print("[Train  ] NLL : {:.6f} || MSE : {:.6f}".format(epoch_ll/num_data, epoch_mse/num_data))
                print("[Eval  ] NLL : {:.6f} || MSE : {:.6f}".format(test_nll, test_mse))
                wandb.log({"train_mse" : (epoch_mse/num_data)}, step=epoch)
                wandb.log({"eval_mse" : test_mse}, step=epoch)    
                
            wandb.log({"train_nll" : (epoch_ll/num_data)}, step=epoch)
            wandb.log({"train_loss" : (epoch_loss/num_data)}, step=epoch)
            wandb.log({"eval_nll" : test_nll}, step=epoch)
            
            if self.task == 'extrapolation' or self.task == 'interpolation':
                print("[Impute ] NLL : {:.6f} || MSE : {:.6f}".format(impute_nll, impute_mse))
                wandb.log({"impute_nll" : impute_nll}, step=epoch)
                wandb.log({"impute_mse" : impute_mse}, step=epoch) 
                
            if (epoch+1) % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                torch.save({
                    'epooch' : epoch,
                    'model_state_dict' : self.dynamics.state_dict()}, 
                           f'./checkpoints/{self.dataset}_{self.task}_{epoch+1}.pt')
                     
    def eval_func(self, dl):
        epoch_mse = 0
        epoch_ll = 0
        
        epoch_impute_mse = 0
        epoch_impute_ll = 0
        
        num_data = 0
        for _, data in enumerate(dl):
        
            if self.dataset == 'pendulum':
                if self.task == 'regression':
                    obs, truth, obs_times, obs_valid = [j.to(self.device).to(torch.float32) for j in data]
                    mask_obs = None
                    mask_truth = None
                else:
                    obs, truth, obs_valid, obs_times, mask_truth = [j.to(self.device).to(torch.float32) for j in data]
                    mask_obs = None
            elif self.dataset == 'person_activity':
                obs = data['inp_obs'].to(self.device)
                truth = data['evd_obs'].to(self.device)
                obs_times = data['inp_tid'].to(self.device)
                labels = data['aux_obs'].to(self.device)
                b, t, _ = obs.size()
                obs_valid = torch.ones(b, t).to(obs.device)
                mask_obs = None
                mask_truth = None
            else:
                obs, truth, obs_valid, obs_times, mask_truth, mask_obs = [j.to(self.device).to(torch.float32) for j in data]
                if self.task == 'extrapolation':
                    obs, obs_valid = adjust_obs_for_extrapolation(obs, obs_valid, obs_times, self.cut_time)

            out, _ = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=32, epoch=None)
            mean, var = out
            
            batch_len = truth.size(0)
            if self.dataset == 'pendulum' and self.task == 'interpolation':
                eval_nll = BNLL_(truth, mean) * batch_len
                eval_mse = MSE_(truth.flatten(start_dim=2), mean.flatten(start_dim=3), mask = mask_truth.flatten(start_dim=2)) * batch_len
            elif self.task == 'classification':
                eval_nll, eval_mse = CNLL_(labels, mean)
            else:
                eval_nll = GNLL_(truth, mean, var, mask=mask_truth) * batch_len
                eval_mse = MSE_(truth, mean, mask=mask_truth) * batch_len
                
            epoch_mse += eval_mse.item()
            epoch_ll += eval_nll.item()
            
            if self.task == 'extrapolation' or self.dataset == 'pendulum':

                if self.dataset == 'pendulum' and self.task == 'interpolation':
                    mask_impute = (1 - obs_valid)[..., None, None, None] * mask_truth
                    impute_nll = BNLL_(truth, mean) * batch_len
                    impute_mse = MSE_(truth.flatten(start_dim=2), mean.flatten(start_dim=3), mask = mask_impute.flatten(start_dim=2)) * batch_len
                elif self.dataset != 'pendulum' and self.task == 'extrapolation':
                    mask_impute = (1 - obs_valid)[..., None] * mask_truth
                    impute_mse = MSE_(truth, mean, mask=mask_impute) * batch_len
                    impute_nll = GNLL_(truth, mean, var, mask=mask_impute) * batch_len
                    
                    epoch_impute_mse += impute_mse.item()
                    epoch_impute_ll += impute_nll.item()

            num_data += batch_len
                
        return epoch_mse/num_data, epoch_ll/num_data, epoch_impute_mse/num_data, epoch_impute_ll/num_data

    def generate_traj(self, data):
        obs, truth, obs_valid, obs_times = [j.to(self.device).to(torch.float32) for j in data]
        mask_obs = None
        out, KL = self.dynamics(obs, obs_times, obs_valid, mask_obs, n_samples=32)
        mean, var = out
        return truth, mean