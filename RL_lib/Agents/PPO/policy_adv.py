"""
    Implements PPO

    PPO: https://arxiv.org/abs/1707.06347
    Modified from policy Written by Patrick Coady (pat-coady.github.io) to implement
    latest version of PPO with pessimistic ratio clipping

    o Has an option to servo both the learning rate and the clip_param to keep KL 
      within  a specified range. This helps on some control tasks
      (i.e., Mujoco Humanid-v2)
 
    o Uses approximate KL 

    o Models distribution of actions as a Gaussian with variance not conditioned on state

    o Has option to discretize sampled actions
 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_utils import Action_converter
import rl_utils
from time import time
import sklearn.utils
 
class Policy(object):
    """ NN-based policy approximation """
    def __init__(self,   net, actions_per_dim=3, kl_targ=0.003,epochs=20,discretize=False, init_func=rl_utils.default_init,
                 test_mode=False,shuffle=True,  servo_kl=False, beta=0.1, max_grad_norm=999, 
                 obs_key='observes', scale_obs=True, verbose=False, rollout_limit=1):
        """
        Args:
            actions_per_dim:        used when discretizing action space
            kl_targ:                target KL divergence between pi_old and pi_new
            epochs:                 number of epochs per update
            discretize:             boolean, True discretizes action space
            test_mode:              boolean, True removes all exploration noise
            shuffle:                boolean, shuffles data each epoch                   
            servo_kl:               boolean:  set to False to not servo beta to KL, which is original PPO implementation
            beta:                   clipping parameter for pessimistic loss ratio
 
        """
        print('Policy with vectorized sample')
        net.apply(init_func)

        self.net = net
        
        self.servo_kl = servo_kl
        self.test_mode = test_mode
        self.discretize = discretize
        self.shuffle = shuffle
        if self.net.recurrent_steps > 1:
            print('Policy: recurrent steps > 1, disabling shuffle')
            self.shuffle = False
        self.actions_per_dim = actions_per_dim
        self.kl_stat = None
        self.entropy_stat = None
        self.kl_targ = kl_targ
        self.epochs = epochs 
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.max_beta = 0.5
        self.min_beta = 0.01 
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.obs_key = obs_key
        self.action_converter = Action_converter(1,actions_per_dim)
        self.grad_monitor = rl_utils.Grad_monitor('Policy', net)
        self.scaler = rl_utils.Scaler(net.obs_dim)
        self.scale_obs = scale_obs
        self.verbose = verbose 
        self.rollout_limit = rollout_limit
        self.rollout_list = []

        if self.net.recurrent_steps > 1:
            self.use_padding = True
        else:
            self.use_padding = False

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.net.lr)

        print('\tTest Mode:         ',self.test_mode)
        print('\tClip Param:        ',self.beta)
        print('\tShuffle :          ',self.shuffle)
        print('\tMax Grad Norm:     ',self.max_grad_norm)
        print('\tRecurrent Steps:   ',self.net.recurrent_steps)
        print('\tRollout Limit:     ',self.rollout_limit)

    def save_params(self,fname):
        fname = 'policy_' + fname + '.pt'
        param_dict = {}
        param_dict['scaler_u']   = self.scaler.means
        param_dict['scaler_var']  = self.scaler.vars
        param_dict['net_state'] = self.net.state_dict()
        torch.save(param_dict, fname)

    def load_params(self,fname):
        fname = 'policy_' + fname + '.pt'
        param_dict = torch.load(fname)
        self.scaler.means = param_dict['scaler_u']
        self.scaler.vars = param_dict['scaler_var']
        self.net.load_state_dict(param_dict['net_state'])

    def _kl_entropy(self, logp, old_logp, log_vars, masks):
        
        """
        We do approximate KL here
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """

        if self.use_padding:
            logp, old_logp = rl_utils.unpad_list([logp, old_logp],masks)

        kl = 0.5 * np.mean((logp - old_logp)**2)
        entropy = 0.5 * (self.net.act_dim * (np.log(2 * np.pi) + 1) +
                              np.sum(log_vars))

        return kl, entropy


    def sample(self, obs, state):
        """Draw sample from policy distribution"""

        if self.scale_obs:
            obs = self.scaler.apply(obs)
        deterministic_action, log_vars, state = self.net.forward(obs, state, np.ones(1), np.zeros(1), return_tensor=False)
        if self.test_mode:
            action = deterministic_action 
        else:
            sd = np.exp(log_vars / 2.0)
            action = deterministic_action + np.random.normal(scale=sd, size=(obs.shape[0], sd.shape[0]))

        if self.discretize:
            idx = self.action_converter.action2idx(action[0])
            discrete_action = self.action_converter.idx2action(idx)
            env_action = discrete_action
        else:  
            env_action = action
        return action, env_action, state 


    def update_scalers(self, rollouts):
        self.scaler.update(rollouts[self.obs_key])

    def update(self, rollouts, logger):
        if len(self.rollout_list) == self.rollout_limit:
            del self.rollout_list[0]
        self.rollout_list.append(rollouts)
        keys = self.rollout_list[0].keys()
        comb_rollouts = {}
        for k in keys:
            comb_rollouts[k] = np.concatenate([r[k] for r in self.rollout_list])
        self.update1(comb_rollouts, logger)
 
    def update1(self, rollouts, logger):
      
        if self.use_padding:
            key = 'padded_'
        else:
            key = '' 
        observes    = rollouts[key + self.obs_key]
        actions     = rollouts[key + 'actions']
        advantages  = rollouts[key + 'advantages']
        states      = rollouts[key + 'policy_states']
        masks       = rollouts[key + 'masks']
        flags       = rollouts[key + 'flags']

        if self.scale_obs:
            observes = self.scaler.apply(observes)
      
 
        actions_pt = torch.from_numpy(actions).float()
        with torch.no_grad():
            means_pt, logvars_pt,  _ = self.net.forward(observes,  states, masks, flags)

        old_logp_pt = self.calc_logp(actions_pt, means_pt, logvars_pt)   
        old_logp = old_logp_pt.detach().numpy() 
        loss, kl, entropy = 0, 0, 0

        advantages_unp = rollouts['advantages'] 
        u_adv = advantages_unp.mean()
        std_adv = advantages_unp.std() +  1e-6

        advantages = (advantages - u_adv) / std_adv 

        t0 = time()
        for e in range(self.epochs):

            if self.shuffle:
                    observes, actions, advantages, states, masks, flags, old_logp = \
                            sklearn.utils.shuffle(observes, actions, advantages, states, masks, flags, old_logp)

            actions_pt = torch.from_numpy(actions).float()

            self.optimizer.zero_grad()
            means_pt, log_vars_pt,  _ = self.net.forward(observes,  states, masks, flags, unroll=True)
            logp_pt = self.calc_logp(actions_pt, means_pt, log_vars_pt)
            loss = self.calc_loss(logp_pt, torch.from_numpy(old_logp).float(), torch.from_numpy(advantages).float(), self.beta, masks)
            loss.backward()
            if self.max_grad_norm is not None:
                ng = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            else:
                ng = None
            self.optimizer.step()
            self.grad_monitor.add(ng)
 
            log_vars = log_vars_pt.detach().numpy()
            kl, entropy = self._kl_entropy(logp_pt.detach().numpy(), old_logp, log_vars, masks)

            if kl > 4.0 * self.kl_targ and self.servo_kl:
                print(' *** BROKE ***')
                break 

        t1 = time()
            
        if self.servo_kl:
            self.adjust_beta(kl)

        for g in self.optimizer.param_groups:
            g['lr'] = self.net.lr * self.lr_multiplier
        self.kl_stat = kl
        self.entropy_stat = entropy
        var_monitor = np.exp(log_vars/2.0)
        self.grad_monitor.show()

        if self.verbose:
            print('POLICY ROLLOUT LIST: ',len(self.rollout_list))
            print('POLICY Update: ',t1-t0,observes.shape)
            print('kl = ',kl, ' beta = ',self.beta,' lr_mult = ',self.lr_multiplier)
            print('var: ' ,var_monitor)
            print('u_adv: ',u_adv)
            print('std_adv: ',std_adv)

        logger.log({'PolicyLoss': loss,
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    'Variance' : np.max(var_monitor),
                    'lr_multiplier': self.lr_multiplier})

    def adjust_beta(self,kl):
        if  kl < self.kl_targ / 2:
            self.beta = np.minimum(self.max_beta, 1.5 * self.beta)  # max clip beta
            #print('too low')
            if self.beta > (self.max_beta/2) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
        elif kl > self.kl_targ * 2:
            #print('too high')
            self.beta = np.maximum(self.min_beta, self.beta / 1.5)  # min clip beta
            if self.beta <= (2*self.min_beta) and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5

    def calc_loss(self,logp, old_logp, advantages, beta, masks):
        if self.use_padding:
            logp, old_logp, advantages = rl_utils.unpad_list([logp, old_logp, advantages], masks)

        ratio = torch.exp(logp - old_logp)
        surr1 = advantages * ratio
        surr2 = advantages * torch.clamp(ratio, 1.0 - beta, 1.0 + beta)
        
        loss = -torch.mean(torch.min(surr1,surr2)) 
        return loss


    def calc_logp(self, act, means, log_vars):
        logp1 = -0.5 * torch.sum(log_vars)
        diff = act - means
        logp2 = -0.5 * torch.sum(torch.mul(diff, diff) / torch.exp(log_vars), 1)
        logp3 = -0.5 * np.log(2.0 * np.pi) * self.net.act_dim
        logp = logp1 + logp2 + logp3
        return logp

