import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rl_utils
from time import time

class MLP1(nn.Module):
    def __init__(self, obs_dim, act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLP1, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt 
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLP1_lvs(nn.Module):
    def __init__(self, obs_dim, act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, lvs_mult=10):
        super(MLP1_lvs, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (lvs_mult * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state


class MLP2_lvs(nn.Module):
    def __init__(self, obs_dim, act_dim, network_scale=1, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, lvs_mult=10):
        super(MLP2_lvs, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid_size = obs_dim * network_scale
        self.lr = base_lr / np.sqrt(hid_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (lvs_mult * hid_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLP3_lvs(nn.Module):
    def __init__(self, obs_dim, act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, lvs_mult=10):
        super(MLP3_lvs, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid2_size = 10 * act_dim 
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_size = int(np.sqrt(hid1_size * hid2_size))
        logvar_speed = (lvs_mult * logvar_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLPS1(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLPS1, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = state_dim 
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(state_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))
        print('MLPS1: ', self.fc1)

    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(s).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLPS2(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLPS2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size+state_dim, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

        print('MLPS2: ', self.fc1)

    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        s = torch.from_numpy(s).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.cat( (x,s) ,dim=1)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLPS3(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLPS3, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size+state_dim, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        s = torch.from_numpy(s).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = torch.cat( (x,s) ,dim=1)
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLPS4(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLPS4, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc01 = nn.Linear(state_dim, state_dim)
        self.fc02 = nn.Linear(state_dim, state_dim//2)

        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size+state_dim//2, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        s = torch.from_numpy(s).float()
        s = self.activation(self.fc01(s))
        s = self.activation(self.fc02(s))

        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.cat( (x,s) ,dim=1)
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLPS5(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLPS5, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fc1 = nn.Linear(obs_dim+state_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        s = torch.from_numpy(s).float()
        x = torch.from_numpy(x).float()
        x = torch.cat( (x,s) ,dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

class MLPS6(nn.Module):
    def __init__(self, obs_dim, act_dim, state_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(MLPS6, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.recurrent_steps = recurrent_steps
        self.fcs1 = nn.Linear(state_dim, state_dim//3)
        self.fcs2 = nn.Linear(state_dim//3, state_dim//9)

        self.fc1 = nn.Linear(obs_dim+state_dim//9, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x, s, masks, flags, return_tensor=True, unroll=None):
        masks = torch.from_numpy(masks).float()
        s = torch.from_numpy(s).float()
        x = torch.from_numpy(x).float()
        s = self.activation(self.fcs1(s))
        s = self.activation(self.fcs2(s))
        x = torch.cat( (x,s) ,dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state


 
class GRU(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None 
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()


class CNNGRU1(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, 
                 recurrent_steps=1, cell=nn.GRUCell):
        super(CNNGRU1, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = 512
        hid3_size = network_scale*act_dim
        hid2_size = 128  #np.sqrt(hid1_size * hid3_size)
        self.lr = base_lr / np.sqrt(hid2_size)
        self.cnn1 = nn.Conv2d(1 , 32, 8, stride=4)
        self.cnn2 = nn.Conv2d(32, 64, 4, stride=2)
        self.cnn3 = nn.Conv2d(64, 32, 3, stride=1)
                
        self.fc1 = nn.Linear(32*8*8, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        #print(x.shape)
        x = self.flatten(x) 
        x = self.activation(self.fc1(x))
        
        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.fc3(r)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class CNNGRU2(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4,
                 recurrent_steps=1, cell=nn.GRUCell):
        super(CNNGRU2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = 128
        hid3_size = network_scale*act_dim
        hid2_size = 64  #np.sqrt(hid1_size * hid3_size)
        self.lr = base_lr / np.sqrt(hid2_size)
        self.cnn1 = nn.Conv2d(1 , 8, 8, stride=4)
        self.cnn2 = nn.Conv2d(8, 16, 4, stride=2)
        self.cnn3 = nn.Conv2d(16, 8, 3, stride=1)
    
        self.fc1 = nn.Linear(8*8*8, hid1_size)
        #self.fc1 = nn.Linear(128, hid1_size)

        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):
        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        #print(x.shape)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.fc3(r)

        log_vars = torch.sum(self.log_vars, 0) - 1.0

        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class CNNGRU4(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4,
                 recurrent_steps=1, cell=nn.GRUCell, input_channels=1):
        super(CNNGRU4, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = 128
        hid3_size = network_scale*act_dim
        hid2_size = 64  #np.sqrt(hid1_size * hid3_size)
        self.lr = base_lr / np.sqrt(hid2_size)
        C = 2
        self.cnn1 = nn.Conv2d(input_channels , C, 8, stride=4)
        self.cnn2 = nn.Conv2d(C, C, 4, stride=2)
        self.cnn3 = nn.Conv2d(C, C, 3, stride=1)
   
        self.fc1 = nn.Linear(128, hid1_size)
        #self.fc1 = nn.Linear(128, hid1_size)

        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):
        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        #print(x.shape)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.fc3(r)

        log_vars = torch.sum(self.log_vars, 0) - 1.0

        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class CNNGRU3(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4,
                 recurrent_steps=1, cell=nn.GRUCell):
        super(CNNGRU3, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        #hid1_size = 128
        #hid3_size = network_scale*act_dim
        #hid2_size = 64  #np.sqrt(hid1_size * hid3_size)
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.lr = base_lr / np.sqrt(hid2_size)
        ch = 4
        self.cnn1 = nn.Conv2d(1 , ch, 8, stride=4)
        self.cnn2 = nn.Conv2d(ch, ch, 4, stride=2)
        self.cnn3 = nn.Conv2d(ch, ch, 3, stride=1)
        self.cnn4 = nn.Conv2d(ch, ch, 3, stride=1)

        #self.fc1 = nn.Linear(8*8*8, hid1_size)
        self.rnn2 = cell(144, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def flatten(self, x):
        return x.view(x.size(0), -1)

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):
        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))

        #print(x.shape)
        x = self.flatten(x)
        #x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.fc3(r)

        log_vars = torch.sum(self.log_vars, 0) - 1.0

        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class GRU_lvs(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell, lvs_mult=10):
        super(GRU_lvs, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (lvs_mult * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class GRU_wn(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU_wn, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.utils.weight_norm(nn.Linear(obs_dim, hid1_size))
        self.rnn2 = nn.utils.weight_norm(nn.utils.weight_norm(cell(hid1_size, hid2_size),name='weight_ih'), name='weight_hh')
        self.fc3 = nn.utils.weight_norm(nn.Linear(hid2_size, hid3_size))
        self.fc4 = nn.utils.weight_norm(nn.Linear(hid3_size, act_dim))

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class GRU2X(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU2X, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = hid1_size 
        self.lr = base_lr / np.sqrt(hid2_size)
        self.rnn1 = cell(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,2,hid1_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        s1 = states[:,0,:]
        s2 = states[:,1,:]

        x = torch.from_numpy(obs).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s1 = rl_utils.batch2seq(s1,self.recurrent_steps)  # T=0 states from rollouts
            s1 = s1[0]
            s2 = rl_utils.batch2seq(s2,self.recurrent_steps)  # T=0 states from rollouts
            s2 = s2[0]
            outputs1 = []
            outputs2 = []
            for i in range(self.recurrent_steps):
                s1 = self.rnn1(x[i], s1 * masks[i])
                outputs1.append(s1)
                s2 = self.rnn2(s1, s2 * masks[i])
                outputs2.append(s2)
            r = torch.stack(outputs2,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            s1 = self.rnn1(x,s1)
            s2 = self.rnn2(s1,s2)
            r = s2
            if s2.shape[0] == 1: 
                states[0,0,:] = s1.detach().numpy()
                states[0,1,:] = s2.detach().numpy()
            else:
                states = None
        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), states

class GRU2X2(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU2X2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        self.hid1_size = hid1_size
        hid3_size = act_dim * network_scale
        hid2_size = hid1_size
        self.lr = base_lr / np.sqrt(hid2_size)
        self.rnn1 = cell(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,2*hid1_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        s1 = states[:,0:self.hid1_size]
        s2 = states[:,self.hid1_size:]

        x = torch.from_numpy(obs).float()
        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s1 = rl_utils.batch2seq(s1,self.recurrent_steps)  # T=0 states from rollouts
            s1 = s1[0]
            s2 = rl_utils.batch2seq(s2,self.recurrent_steps)  # T=0 states from rollouts
            s2 = s2[0]
            outputs1 = []
            outputs2 = []
            for i in range(self.recurrent_steps):
                s1 = self.rnn1(x[i], s1 * masks[i])
                outputs1.append(s1)
                s2 = self.rnn2(s1, s2 * masks[i])
                outputs2.append(s2)
            r = torch.stack(outputs2,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            s1 = self.rnn1(x,s1)
            s2 = self.rnn2(s1,s2)
            r = s2
            states = torch.cat((s1,s2),dim=1)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0

        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), states.detach().numpy()

class GRUI(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRUI, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = IRNNCell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class GRUI2(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRUI2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = IRNNCell2(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, s)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()


class GRU2(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1, cell=nn.GRUCell):
        super(GRU2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = cell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        states = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll and self.recurrent_steps > 1:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            states = rl_utils.batch2seq(states,self.recurrent_steps)  # T=0 states from rollouts
            s = states[0]
            outputs = []
            for i in range(self.recurrent_steps):
                s = self.rnn2(x[i], s * masks[i])
                outputs.append(s)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            r = s = self.rnn2(x, states)

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, None
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()

class LSTM(nn.Module):
    def __init__(self, obs_dim,  act_dim, network_scale=10, activation=torch.tanh, base_lr=9e-4, recurrent_steps=1):
        super(LSTM, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.recurrent_steps = recurrent_steps
        hid1_size = obs_dim * network_scale
        hid3_size = act_dim * network_scale
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        self.lr = base_lr / np.sqrt(hid2_size)
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.rnn2 = nn.LSTMCell(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, hid3_size)
        self.fc4 = nn.Linear(hid3_size, act_dim)

        self.initial_state = np.zeros((1,2*hid2_size))
        self.initial_error = np.zeros((1,obs_dim))

        logvar_speed = (10 * hid3_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))

    def forward(self, obs,  states,  masks, flags, return_tensor=True,unroll=False):

        masks = torch.from_numpy(masks).float()

        x = torch.from_numpy(obs).float()
        s = torch.from_numpy(states).float()

        x = self.activation(self.fc1(x))

        if unroll:
            x = rl_utils.batch2seq(x,self.recurrent_steps)
            masks = rl_utils.batch2seq(masks,self.recurrent_steps)
            s = rl_utils.batch2seq(s,self.recurrent_steps)  # T=0 states from rollouts
            s = s[0]
            outputs = []
            for i in range(self.recurrent_steps):
                h,c = self.rnn2(x[i], torch.split(s * masks[i], s.size(1)//2, dim=1))
                s = torch.cat([h,c],dim=1)
                outputs.append(h)
            r = torch.stack(outputs,dim=0)
            r = rl_utils.seq2batch(r,self.recurrent_steps)
        else:
            h,c = torch.split(s, s.size(1)//2, dim=1 )
            h, c = self.rnn2(x, (h, c))
            s = torch.cat([h,c],dim=1)
            r = h 

        x = self.activation(self.fc3(r))
        x = self.fc4(x)

        log_vars = torch.sum(self.log_vars, 0) - 1.0


        if return_tensor:
            return x, log_vars, s
        else:
            return x.detach().numpy(),  log_vars.detach().numpy(), s.detach().numpy()


class MLP2(nn.Module):
    def __init__(self, obs_dim, act_dim, network_scale=10, activation=torch.tanh):
        super(MLP2, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.activation = activation
        self.activation = activation
        hid_size = 10*obs_dim
        self.lr = 9e-4 / np.sqrt(hid_size)

        self.fc1 = nn.Linear(obs_dim, hid_size)
        self.fc2 = nn.Linear(hid_size, hid_size)
        self.fc3 = nn.Linear(hid_size, hid_size)
        self.fc4 = nn.Linear(hid_size, act_dim)

        self.initial_state    = np.zeros((1,1))
        self.initial_state_pt = torch.from_numpy(np.zeros((1,1)))

        logvar_speed = (10 * hid_size) // 48
        self.log_vars = nn.Parameter(torch.zeros(logvar_speed, act_dim))


    def forward(self, x,  s, masks, flags, return_tensor=True):
        masks = torch.from_numpy(masks).float()
        x = torch.from_numpy(x).float()
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        log_vars = torch.sum(self.log_vars, 0) - 1.0
        if return_tensor:
            return x, log_vars, self.initial_state_pt
        else:
            return x.detach().numpy(), log_vars.detach().numpy(), self.initial_state

    

