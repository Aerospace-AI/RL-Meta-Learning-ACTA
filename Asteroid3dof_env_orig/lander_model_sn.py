import numpy as np
import env_utils as envu

class Lander_model(object):

    def __init__(self, altimeter=None, use_trajectory_list=False, divert = (0.0,  0.0, 0.0), sensor_bias_range=(0.0,0.0), sensor_sd=0.0, 
                 apf_v0=100, apf_tau1=20, apf_tau2=1, gain=1.5, apf_vf1=-2.0, apf_vf2=-1.0, apf_atarg=5.0): 
        self.traj = {}
        self.state_keys = ['position','velocity','thrust','mass','t_go']
        self.trajectory_list = []
        self.trajectory = {} 
        self.max_thrust = 2 
        self.min_thrust = 0
        self.init_mass = 500.
        self.dry_mass = 200 
        self.use_trajectory_list = use_trajectory_list        

        self.sensor_bias_range=sensor_bias_range
        self.sensor_sd = sensor_sd

        self.apf_v0 = apf_v0
        self.apf_vf1 =  np.zeros(3)
        self.apf_vf1[2] = apf_vf1
        self.apf_vf2 =  np.zeros(3)
        self.apf_vf2[2] = apf_vf2
        self.apf_tau1 = apf_tau1
        self.apf_tau2 = apf_tau2
        self.apf_atarg = apf_atarg

        self.divert = divert
        self.apf_state = False

        self.altimeter = altimeter 

        assert self.apf_vf1.shape[0] == 3
        assert self.apf_vf2.shape[0] == 3

        self.track_func         = self.track_func1
        self.apf_pot            = self.apf_pot1
        self.get_state_agent    = self.get_state_agent2

        self.trajectory = {}
        self.state = {}
        self.state['t_go'] = None
        self.prev_state = {}
        print('lander model apf')

    def clear_trajectory(self):
        for k in self.get_engagement_keys():
            self.trajectory[k] = []
        self.sensor_bias = np.random.uniform(self.sensor_bias_range[0],self.sensor_bias_range[1])
 
    def update_trajectory(self, done, t, action):

        es = self.get_landing_state(t)
        for k,v in es.items():
            self.trajectory[k].append(v)
         
        if(done):
            if self.use_trajectory_list:
                self.trajectory_list.append(self.trajectory.copy())

    def get_state_agent_po(self,t):
        return self.state['position'] 

    def get_state_agent_po_vc(self,t):
        vc = envu.get_vc(self.state['position'], self.state['velocity'])
        state = np.hstack((self.state['position'], vc)) 
        return state 

    def get_state_agent_po_delta(self,t):
        position =  self.state['position']
        delta = self.state['position'] - self.prev_state['position']
        state = np.hstack((position,delta))
        return state 

    def get_state_agent_vo(self,t):
        return self.state['velocity']

    def get_state_agent_eo(self,t):
        error, t_go = self.track_func(self.state['position'], self.state['velocity'])
        self.state['t_go'] = t_go
        return error

    def get_state_agent_eo_range(self,t):
        error, t_go = self.track_func(self.state['position'], self.state['velocity'])
        range = np.linalg.norm(self.state['position'])
        state = np.hstack((error, range))
        self.state['t_go'] = t_go
        return state

    def get_state_agent0(self,t):
        rg = np.linalg.norm(self.state['position'])
        vg = np.linalg.norm(self.state['velocity'])
        t_go = rg / vg
        state = np.hstack((self.state['position'],t_go))
        return state

    def get_state_agent_fullref(self,t):
        vtarg , t_go = self.apf_pot(np.hstack((self.state['position'], self.state['velocity'])))
        state = np.hstack((self.state['position'], self.state['velocity'], vtarg, t_go))
        return state

    def get_state_agent1(self,t):
        position = self.add_noise(self.state['position'])
        velocity = self.add_noise(self.state['velocity'])

        state = np.hstack((position, velocity))
        return state

    def get_state_agent2(self,t):
        error, t_go = self.track_func(self.state['position'], self.state['velocity'])
        state = np.hstack((error, t_go))
        self.state['t_go'] = t_go
        return state 

    def get_state_agent4(self,t):
        error, t_go = self.track_func(self.state['position'], self.state['velocity'])
        state = np.hstack((error, t_go, self.apf_state))
        self.state['t_go'] = t_go
        return state

    def get_state_agent5(self,t):
        error, t_go = self.track_func(self.state['position'], self.state['velocity'])
        state = np.hstack((error, t_go, self.state['position'][2]))
        self.state['t_go'] = t_go
        return state

    def get_state_agent_dtm(self,t):
        alt,vc = self.altimeter.get_reading( self.state['position'] , self.state['velocity'])
        return np.hstack((alt,vc))

    def get_state_agent_dtm_gtv(self,t):
        alt,vc = self.altimeter.get_reading( self.state['position'] , self.state['velocity'])
        return np.hstack((alt,self.state['velocity']))

    def get_state_agent_dtm_altonly(self,t):
        alt,vc = self.altimeter.get_reading( self.state['position'] , self.state['velocity'])
        return alt

    def get_state_agent_eo_done(self,t):
        done = self.state['position'][2] < 0
        error, t_go = self.track_func(self.state['position'], self.state['velocity'])
        self.state['t_go'] = t_go
        state = np.hstack((error, done))
        return state

    def get_state_dynamics(self):
        state = np.hstack((self.state['position'], self.state['velocity'], self.state['mass']))
        return state
 
    def show_cum_stats(self):
        print('Cumulative Stats (mean,std,max,argmax)')
        stats = {}
        argmax_stats = {}
        keys = ['thrust','glideslope','sc_margin']
        formats = {'thrust' : '{:6.2f}', 'glideslope' : '{:6.3f}', 'sc_margin' :  '{:6.3f}'} 
        for k in keys:
            stats[k] = []
            argmax_stats[k] = []
        for traj in self.trajectory_list:
            for k in keys:
                v = traj[k]
                v = np.asarray(v)
                if len(v.shape) == 1:
                    v = np.expand_dims(v,axis=1)
                wc = np.max(np.linalg.norm(v,axis=1))
                argmax_stats[k].append(wc)
                stats[k].append(np.linalg.norm(v,axis=1))
                 
        for k in keys:
            f = formats[k]
            v = stats[k]
            v = np.concatenate(v)
            #v = np.asarray(v)
            s = '%-8s' % (k)
            #print('foo: ',k,v,v.shape)
            s += envu.print_vector(' |',np.mean(v),f)
            s += envu.print_vector(' |',np.std(v),f)
            s += envu.print_vector(' |',np.min(v),f)
            s += envu.print_vector(' |',np.max(v),f)
            argmax_v = np.asarray(argmax_stats[k])
            s += ' |%6d' % (np.argmax(argmax_v))
            print(s)

    def show_final_stats(self,type='final'):
        if type == 'final':
            print('Final Stats (mean,std,min,max)')
            idx = -1
        else:
            print('Initial Stats (mean,std,min,max)')
            idx = 0
 
        stats = {}
        keys = ['norm_vf', 'norm_rf', 'position', 'velocity', 'fuel', 'glideslope']
        formats = {'norm_rf' : '{:8.1f}' , 'norm_vf' : '{:8.3f}', 'position' : '{:8.1f}' , 'velocity' : '{:8.3f}', 'fuel' : '{:6.2f}', 'glideslope' : '{:6.2f}'}

        for k in keys:
            stats[k] = []
        for traj in self.trajectory_list:
            for k in keys:
                v = traj[k]
                v = np.asarray(v)
                if len(v.shape) == 1:
                    v = np.expand_dims(v,axis=1)
                stats[k].append(v[idx])

        for k in keys:
            f = formats[k]
            v = stats[k]
            s = '%-8s' % (k)
            s += envu.print_vector(' |',np.mean(v,axis=0),f)
            s += envu.print_vector(' |',np.std(v,axis=0),f)
            s += envu.print_vector(' |',np.min(v,axis=0),f)
            s += envu.print_vector(' |',np.max(v,axis=0),f)
            print(s)


    def show_episode(self, idx=0):
        
        traj = self.trajectory_list[idx]
        t = np.asarray(traj['time'])
        p = np.asarray(traj['position'])
        v = np.asarray(traj['velocity'])
        c = np.asarray(traj['thrust'])

        f1 = '{:8.1f}'
        f2 = '{:8.3f}' 
        for i in range(t.shape[0]):

            s = 't: %6.1f' % (t[i])
            s += envu.print_vector(' |',p[i],f1)
            s += envu.print_vector(' |',v[i],f2)
            s += envu.print_vector(' |',c[i],f2)
            print(s)

         
    def get_landing_state(self,t):

        landing_state = {}
        landing_state['t'] = t
        landing_state['position'] = self.state['position'] 
        landing_state['velocity'] = self.state['velocity']
        landing_state['norm_rf'] = np.linalg.norm(self.state['position'])
        landing_state['norm_vf'] = np.linalg.norm(self.state['velocity'])
        landing_state['v_ratio'] = np.linalg.norm(self.state['velocity'][0:2]) / np.abs(self.state['velocity'][2])
        landing_state['thrust'] = self.state['thrust'] 
        landing_state['mass']   = self.state['mass']
        landing_state['fuel']   = self.init_mass-self.state['mass']
        landing_state['vc'] =  envu.get_vc(self.state['position'], self.state['velocity']) 
        landing_state['range'] = np.linalg.norm(self.state['position']) 
        landing_state['los'] = self.state['position'] / np.linalg.norm(self.state['position']) 
        landing_state['dlos'] = envu.get_dlos(self.state['position'], self.state['velocity']) 
        landing_state['zem'] = envu.get_zem(self.state['position'], self.state['velocity'])
        landing_state['t_go'] = self.state['t_go'] 
        return landing_state

    def get_engagement_keys(self):
        keys = ['t', 'norm_rf', 'norm_vf', 'position', 'velocity', 'thrust', 'mass', 'fuel', 'vc', 'range', 'los', 'dlos', 'zem','reward','fuel_reward','landing_reward','tracking_reward','glideslope_reward', 'landing_margin', 'glideslope', 'sc_margin', 'glideslope_penalty','sc_penalty','sc_reward','range_reward','v_ratio', 't_go', 'value']
        return keys


    def show(self):

        """
            For debugging
        """

        print('Position:        ',self.position)
        print('Velocity:        ',self.velocity)
        print('Mass:            ',self.mass)

        
    def track_func1(self,next_pos, next_vel):
        vtarg , t_go = self.apf_pot(np.hstack((next_pos, next_vel)))
        error = next_vel - vtarg
        return error, t_go


    def apf_pot0(self,state):
        rg = state[0:3]
        mag_pot =  self.apf_v0 
        dir_pot = -rg / np.linalg.norm(rg) 

        pot = mag_pot * dir_pot

        return pot, 0.0

    def apf_pot1(self,state):
        rg = state[0:3]
        vg = state[3:6]
        mag_rg = np.linalg.norm(rg)
        mag_vg = np.linalg.norm(vg)

        if rg[2] < self.apf_atarg:
            self.apf_state = True  # sticky

        if rg[2] > self.apf_atarg:
            rg1 = state[0:3] - 1.0*np.asarray([0,0,self.apf_atarg])
            vg1 = state[3:6] - self.apf_vf1
            mag_rg1 = np.linalg.norm(rg1)
            mag_vg1 = np.linalg.norm(vg1)
            #if t_go < 1:
            #    t_go = t_go**3
            tau = self.apf_tau1
        else:
            rg1 = 1.0*np.asarray([0,0,state[2]])
            vg1 = state[3:6] - self.apf_vf2
            mag_vg1 = np.linalg.norm(vg1)
            mag_rg1 = np.linalg.norm(rg1)
            tau = self.apf_tau2

        t_go = mag_rg1 / mag_vg1
        mag_pot =  self.apf_v0 * (1. - np.exp(-t_go / tau))
        dir_pot = -rg1 / mag_rg1

        pot = mag_pot * dir_pot

        return pot, t_go

    def apf_pot2(self,state):
        rg = state[0:3]
        vg = state[3:6]
        mag_rg = np.linalg.norm(rg)
        mag_vg = np.linalg.norm(vg)

        t_go = mag_rg / mag_vg
        if t_go < 1:
            t_go = t_go **3
        mag_pot = self.apf_v0 * (1. - np.exp(-t_go / self.apf_tau1))
        dir_pot = -rg / mag_rg

        pot = mag_pot * dir_pot
        return pot, t_go

    def get_vc(self,rg,vg):
        vc = envu.get_vc(rg,vg)
        return np.maximum(0.1,vc)


    def apf_pot_fuel(self,state):
        xy = np.linalg.norm(state[0:2])
        atarg = self.apf_atarg * (1 - np.exp(-xy/self.apf_tau_xy))
        #print('DEBUG: ',state[0:2], xy, atarg)
        self.atarg_debug = atarg
        rg = state[0:3] - 1.0*np.asarray([0,0,atarg])
        vg = state[3:6] - self.apf_vf1
        mag_rg = np.linalg.norm(rg)
        mag_vg = np.linalg.norm(vg)

        dir_pot = -rg / mag_rg
        t_go = mag_rg / mag_vg

        mag_pot =  self.apf_v0 * (1. - np.exp(-t_go / self.apf_tau1))
        pot = mag_pot * dir_pot

        return pot, t_go

    def apf_pot_divert(self,state):
        if  state[2] < self.divert[2]:
            divert_vec1 = 1.0 * np.asarray([self.divert[0],self.divert[1],self.apf_atarg])
            divert_vec2 = 1.0 * np.asarray([0.0,0.0,state[2]])

        else:
            divert_vec1 =  1.0 * np.asarray([0., 0., self.apf_atarg])
            divert_vec2 =  1.0 * np.asarray([0., 0., state[2]])

        rg = state[0:3]
        vg = state[3:6]
        mag_rg = np.linalg.norm(rg)
        mag_vg = np.linalg.norm(vg)

        if rg[2] < self.apf_atarg:
            self.apf_state = True  # sticky

        if rg[2] > self.apf_atarg:
            rg1 = state[0:3] - divert_vec1
            vg1 = state[3:6] - self.apf_vf1
            mag_rg1 = np.linalg.norm(rg1)
            mag_vg1 = np.linalg.norm(vg1)
            tau = self.apf_tau1
        else:
            rg1 = divert_vec2
            vg1 = state[3:6] - self.apf_vf2
            mag_vg1 = np.linalg.norm(vg1)
            mag_rg1 = np.linalg.norm(rg1)
            tau = self.apf_tau2

        t_go = mag_rg1 / mag_vg1
        mag_pot =  self.apf_v0 * (1. - np.exp(-t_go / tau))
        dir_pot = -rg1 / mag_rg1

        pot = mag_pot * dir_pot

        return pot, t_go

    def add_noise(self,obs):
        #print(self.sensor_bias)
        #print('BEFORE: ', obs)
        bias = self.sensor_bias * np.abs(obs)
        noise = self.sensor_sd * np.abs(obs)
        obs = obs + bias + noise * np.random.normal(size=obs.shape)
        #print('AFTER: ', obs)
        return obs
