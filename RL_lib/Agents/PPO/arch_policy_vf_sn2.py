import numpy as np
import rl_utils

class Arch(object):
    def __init__(self, sensor_bias_range=(0.0,0.0), sensor_sd=0.0):
        self.sensor_bias_range = sensor_bias_range
        self.sensor_sd = sensor_sd

    def run_episode(self, env, policy, val_func, model, recurrent_steps, use_timestep, add_padding=True):
        self.sensor_bias = np.random.uniform(self.sensor_bias_range[0],self.sensor_bias_range[1])

        obs = env.reset()
        obs_n = self.add_noise(obs)
        observes, actions, vpreds, rewards1, rewards2,  nobserves,  policy_states, vf_states   =    [], [], [], [], [], [], [], []
        traj = {}
        done = False
        step = 0.0
        policy_state = policy.net.initial_state
        vf_state = val_func.net.initial_state
        flag = 1
        while not done:


            obs = obs.astype(np.float64).reshape((1, -1))
            policy_states.append(policy_state)
            vf_states.append(vf_state)

            if use_timestep:
                obs = np.append(obs, [[step]], axis=1)  # add time step feature
            observes.append(obs)

            action, env_action, policy_state = policy.sample(obs_n, policy_state)
            actions.append(action)

            vpred, vf_state = val_func.predict(obs_n, vf_state)
            env.lander.trajectory['value'].append(vpred.copy())

            vpreds.append(vpred) 

            obs, reward, done, reward_info = env.step(env_action)
            obs_n = self.add_noise(obs)
            nobserves.append(obs.astype(np.float64).reshape((1, -1)))

            reward1 = reward[0]
            reward2 = reward[1]
            if not isinstance(reward1, float):
                reward1 = np.asscalar(reward1)
            if not isinstance(reward1, float):
                reward2 = np.asscalar(reward2)
            rewards1.append(reward1)
            rewards2.append(reward2)
            step += 1e-3  # increment time step feature
            flag = 0

        env.lander.trajectory['value'].append(vpred.copy())

        traj['observes'] = np.concatenate(observes)
        traj['actions'] = np.concatenate(actions)
        traj['rewards1'] = np.array(rewards1, dtype=np.float64)
        traj['rewards2'] = np.array(rewards2, dtype=np.float64)
        traj['nobserves'] = np.concatenate(nobserves)
        traj['policy_states'] = np.concatenate(policy_states)
        traj['vf_states'] = np.concatenate(vf_states)
        traj['vpreds'] = np.array(vpreds, dtype=np.float64)
        traj['flags'] = np.zeros(len(observes))
        traj['flags'][0] = 1

        traj['masks'] = np.ones_like(traj['rewards1'])

        return traj

    def update_scalers(self, policy, val_func, model, rollouts):
        policy.update_scalers(rollouts)
        val_func.update_scalers(rollouts)
        
    def update(self,policy,val_func,model,rollouts, logger):
        policy.update(rollouts, logger)
        val_func.fit(rollouts, logger) 

    def add_noise(self,obs):
        bias = self.sensor_bias * np.abs(obs)
        noise = self.sensor_sd * np.abs(obs)
        obs = obs + bias + noise * np.random.normal(size=obs.shape)
        return obs.astype(np.float64).reshape((1, -1)) 
