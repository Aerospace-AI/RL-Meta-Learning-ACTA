"""
State-Value Function


"""
import rl_utils
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from time import time
import sklearn.utils

class Value_function(object):
    """ NN-based state-value function """
    def __init__(self,  net, epochs=20, batch_size=256, cliprange=0.5, lr=None,shuffle=True, init_func=rl_utils.default_init,  max_grad_norm=999, 
                 obs_key='observes', scale_obs=True, rollout_limit=1, verbose=False):
        """
        Args:
            obs_dim:        number of dimensions in observation vector (int)
            epochs:         number of epochs per update
            cliprange:      for limiting value function updates
 
        """
        print('Value Funtion')
        net.apply(init_func)
        self.net = net
        if self.net.recurrent_steps > 1:
            self.use_padding = True
        else:
            self.use_padding = False

        self.shuffle = shuffle
        self.batch_size = batch_size

        if self.net.recurrent_steps > 1:
            print('Value Function: recurrent steps > 1, disabling shuffle and batching')
            self.shuffle = False
            self.batch_size = 9999999
        self.cliprange = cliprange
        self.exp_var_stat = None
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.obs_key = obs_key
        self.optimizer = torch.optim.Adam(self.net.parameters(), self.net.lr)  
        self.grad_monitor = rl_utils.Grad_monitor('ValFun',net)
        self.scaler = rl_utils.Scaler(net.obs_dim)
        self.scale_obs = scale_obs
        self.verbose = verbose

        self.rollout_list = []
        self.rollout_limit = rollout_limit

        print('\tClip Range:        ',self.cliprange)
        print('\tShuffle :          ',self.shuffle)
        print('\tBatch Size :       ',self.batch_size)
        print('\tMax Grad Norm:     ',self.max_grad_norm)
        print('\tRecurrent Steps:   ',self.net.recurrent_steps)
        print('\tRollout Limit:     ',self.rollout_limit)

    def save_params(self,fname):
        fname = 'valfunc_' + fname + '.pt'
        param_dict = {}
        param_dict['scaler_u']   = self.scaler.means
        param_dict['scaler_var']  = self.scaler.vars
        param_dict['net_state'] = self.net.state_dict()
        torch.save(param_dict, fname)

    def load_params(self,fname):
        fname = 'valfunc_' + fname + '.pt'
        param_dict = torch.load(fname)
        self.scaler.means = param_dict['scaler_u']
        self.scaler.vars = param_dict['scaler_var']
        self.net.load_state_dict(param_dict['net_state'])

    def update_scalers(self, rollouts):
        self.scaler.update(rollouts[self.obs_key])

    def fit(self, rollouts, logger):
        if len(self.rollout_list) == self.rollout_limit:
            del self.rollout_list[0]
        self.rollout_list.append(rollouts)
        keys = self.rollout_list[0].keys()
        comb_rollouts = {}
        for k in keys:
            comb_rollouts[k] = np.concatenate([r[k] for r in self.rollout_list])
        self.fit1(comb_rollouts, logger)
 
    def fit1(self, rollouts,  logger):
        if self.use_padding:
            key = "padded_"
        else:
            key = ""
        observes = rollouts[key + self.obs_key]
        vtarg = rollouts[key + 'disc_sum_rew']
        masks = rollouts[key + 'masks']
        flags = rollouts[key + 'flags']
        states = rollouts[key + 'vf_states']

        if self.scale_obs:
            observes = self.scaler.apply(observes)

        t0 = time()

        old_vpred, _ = self.net.forward(observes, states, masks, flags, return_tensor=False)  # check explained variance prior to update
        old_exp_var = rl_utils.calc_exp_var(vtarg, old_vpred, masks, self.use_padding)
        indices = rl_utils.get_mini_ids(observes.shape[0], self.batch_size)
        for e in range(self.epochs):
            if self.shuffle:
                observes, vtarg,  old_vpred, states, masks, flags = sklearn.utils.shuffle(observes, vtarg, old_vpred, states, masks, flags)
            for j in range(len(indices)):
                start = indices[j][0]
                end   = indices[j][1] 
                self.optimizer.zero_grad()
                vpred_pt, _ = self.net.forward(observes[start:end, :],  states[start:end, :], masks[start:end], flags[start:end], unroll=True)
                loss = self.get_loss(vpred_pt, torch.from_numpy(old_vpred[start:end]).float(), torch.from_numpy(vtarg[start:end]).float(), masks[start:end])
                loss.backward()
                if self.max_grad_norm is not None:
                    ng=nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                else:
                    ng=None
                self.grad_monitor.add(ng)
                self.optimizer.step()
        y_hat, _ = self.net.forward(observes,states, masks, flags, return_tensor=False)
        ev_loss = np.mean(np.square(y_hat - vtarg))         # explained variance after update
        exp_var = rl_utils.calc_exp_var(vtarg, y_hat, masks, self.use_padding)

        self.exp_var_stat = exp_var
        self.grad_monitor.show()
        logger.log({'ValFuncLoss': ev_loss,
                    'ExplainedVarNew': exp_var,
                    'ExplainedVarOld': old_exp_var})

        t1 = time()
        if self.verbose:
            print('VF MODEL ROLLOUT LIST: ',len(self.rollout_list))
            print('VF FIT: ',t1-t0,observes.shape)
            print('VF FIT: ',np.max(np.abs(vtarg)), np.max(np.abs(observes)), np.mean(np.abs(vtarg)), np.mean(np.abs(observes)))
            print('VF SCALE: ', self.scaler.means, 3./np.sqrt(self.scaler.vars))

    def get_loss(self, pred, old_pred, targ, masks):
        if self.use_padding:
            pred, old_pred, targ = rl_utils.unpad_list([pred, old_pred, targ],masks)
        if self.cliprange is not None:
            vpred_clipped = old_pred + torch.clamp(pred - old_pred, -self.cliprange, self.cliprange)
            error = pred - targ
            loss1 = torch.mul(error, error)
            error = vpred_clipped - targ 
            loss2 = torch.mul(error, error)
            loss = 0.5 * torch.mean(torch.max(loss1,loss2))
        else:
            error = pred - targ
            loss = 0.5 * torch.mean(torch.mul(error, error))
        return loss

    def predict(self,x, s):
        if self.scale_obs:
            x = self.scaler.apply(x)
        y, s = self.net.forward(x, s, np.asarray([1.0]),  np.asarray([0.0]), return_tensor=False)
        return y, s 
     

