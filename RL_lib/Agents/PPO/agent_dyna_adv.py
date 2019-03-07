import scipy.signal
import signal
import numpy as np
import rl_utils

"""


"""

class Agent(object):
    def __init__(self, arch, policy, val_func, model, dyna, env,  logger, policy_episodes=20, policy_steps=10, gamma1=0.0, gamma2=9.995, lam=0.98,
                    use_timestep=False, monitor=None, recurrent_steps=1, warmup_updates=10, dyna_stop=150):
        self.arch = arch
        self.env = env
        self.monitor = monitor
        self.policy_steps = policy_steps
        self.logger = logger
        self.use_timestep = use_timestep
        self.policy = policy 
        self.val_func = val_func
        self.model = model
        self.dyna = dyna
        self.warmup_updates = warmup_updates
        self.dyna_stop = dyna_stop
        self.policy_episodes = policy_episodes
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        self.lam = lam
        self.global_steps = 0
        self.recurrent_steps = recurrent_steps
        self.update_cnt = 0

        self.dyna.gamma = gamma1
        print('Agent: ', dyna.gamma, gamma1)
        
        """ 

            Args:
                policy:                 policy object with update() and sample() methods
                val_func:               value function object with fit() and predict() methods
                env:                    environment
                logger:                 Logger object

                policy_episodes:        number of episodes collected before update
                policy_steps:           minimum number of steps before update
                    (will update when either episodes > policy_episodes or steps > policy_steps)

                gamma:                  discount rate
                lam:                    lambda for GAE calculation
                use_timestep:           boolean, True enables time step feature which sometimes works better than a 
                                        low discount rate for continuing tasks with per-step rewards (like Mujoco envs)
                monitor:                A monitor object like RL_stats to plot interesting stats as learning progresses
                                        Monitor object implements update_episode() and show() methods 

        """ 
  
    def run_episode(self):
        traj = self.arch.run_episode(self.env, self.policy, self.val_func, self.model, self.recurrent_steps, self.use_timestep)
        padded_traj = {}
        for k,v in traj.items():
            key = 'padded_' +  k
            padded_traj[key], mask = rl_utils.add_padding(traj[k], self.recurrent_steps)

        traj.update(padded_traj)

        return traj

    def run_policy(self,episode_cnt,warmup=False):
        """ Run policy and collect data for a minimum of min_steps and min_episodes
        Args:
            episode_cnt: current episode number, used for logging stats     

        Returns: list of trajectory dictionaries, list length = number of episodes
            'observes' : NumPy array of states from episode
            'actions' : NumPy array of actions from episode
            'rewards' : NumPy array of (un-discounted) rewards from episode
        """
        total_steps = 0
        e_cnt = 0
        trajectories = []
        while e_cnt <= self.policy_episodes or total_steps < self.policy_steps:
            traj  = self.run_episode()
            if self.monitor is not None and not warmup:
                self.monitor.update_episode(np.sum(traj['rewards1']) + np.sum(traj['rewards2']),  traj['observes'].shape[0])
            total_steps += traj['observes'].shape[0]
            trajectories.append(traj)
            e_cnt += 1
        self.add_disc_sum_rew(trajectories, self.gamma1, self.gamma2)  # calculated discounted sum of Rs
        # concatenate all episodes into single NumPy arrays
        keys = trajectories[0].keys()
        if 'vpreds' not in keys and self.val_func is not None:
            print('vpred not found in keys, adding')
            self.add_vpreds(trajectories) 
        rollouts = {}
        for k in keys:
            rollouts[k] = np.concatenate([t[k] for t in trajectories])

        rollouts['padded_advantages'] = rollouts['padded_disc_sum_rew'] - rollouts['padded_vpreds']
        rollouts['advantages']        = rl_utils.unpad(rollouts['padded_advantages'], rollouts['masks'])

        self.dyna.add_rollouts(rollouts)
        self.arch.update_scalers(self.policy, self.val_func, self.model, rollouts)

        if not warmup:
            self.arch.update(self.policy, self.val_func, self.model, rollouts, self.logger)
            self.log_batch_stats(rollouts['observes'], rollouts['actions'],  rollouts['disc_sum_rew'], episode_cnt)
            self.global_steps += total_steps
            self.logger.log({'_MeanReward': np.mean([t['rewards1'].sum() + t['rewards2'].sum() for t in trajectories]),
                         '_StdReward': np.std([t['rewards1'].sum() + t['rewards2'].sum() for t in trajectories]),
                         '_MinReward': np.min([t['rewards1'].sum() + t['rewards2'].sum()  for t in trajectories]),
                         'Steps': total_steps,
                         'TotalSteps' : self.global_steps})
            if self.monitor is not None: 
                self.monitor.show()
            if self.update_cnt > self.warmup_updates and self.update_cnt < self.dyna_stop:
                sim_rollouts = self.dyna.simulate(self.model, self.policy, self.val_func)
                self.policy.update(sim_rollouts, self.logger)
            self.update_cnt += 1
  
        return trajectories

    def train(self,train_episodes, train_samples=None, warmup_updates=1):
        for i in range(warmup_updates):
            _ = self.run_policy(-1,warmup=True)
        print('*** SCALER WARMUP COMPLETE *** ')
        episode = 0
       
        if train_samples is not None:
            while self.global_steps < train_samples:
                trajectories = self.run_policy(episode)
                self.logger.write(display=True)
                episode += len(trajectories)
        else: 
            while episode < train_episodes: 
                trajectories = self.run_policy(episode)
                self.logger.write(display=True)
                episode += len(trajectories)
            
  
    def discount(self,x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


    def add_disc_sum_rew(self,trajectories, gamma1, gamma2):
        """ Adds discounted sum of rewards to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            gamma: discount

        Returns:
            None (mutates trajectories dictionary to add 'disc_sum_rew')
        """
        for trajectory in trajectories:
            if gamma1 < 0.999:  # don't scale for gamma ~= 1
                rewards1 = trajectory['rewards1'] * (1 - gamma1)
            else:
                rewards1 = trajectory['rewards1'] * (1-0.999)

            if gamma2 < 0.999:  # don't scale for gamma ~= 1
                rewards2 = trajectory['rewards2'] * (1 - gamma2)
            else:
                rewards2 = trajectory['rewards2'] * (1-0.999)

            disc_sum_rew1 = self.discount(rewards1, gamma1)
            disc_sum_rew2 = self.discount(rewards2, gamma2)

            trajectory['disc_sum_rew'] = disc_sum_rew1 + disc_sum_rew2
            trajectory['advantages'] =  trajectory['disc_sum_rew'] - trajectory['vpreds']
            trajectory['padded_disc_sum_rew'], _ = rl_utils.add_padding(trajectory['disc_sum_rew'], self.recurrent_steps)
            trajectory['padded_advantages'], _ =   rl_utils.add_padding(trajectory['advantages'], self.recurrent_steps)
  
    def add_vpreds(self,trajectories):
        """ Adds estimated value to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()
            val_func: object with predict() method, takes observations
                and returns predicted state value

        Returns:
            None (mutates trajectories dictionary to add 'values')
        """
        for trajectory in trajectories:
            observes = trajectory['observes']
            values, _ = self.val_func.predict(observes, np.zeros_like(self.val_func.net.initial_state))
            trajectory['vpreds'] = values

    def log_batch_stats(self,observes, actions, disc_sum_rew, episode):
        """ Log various batch statistics """
        self.logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
            })

