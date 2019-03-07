import numpy as np
import rl_utils

class Arch(object):
    def __init__(self,   zero_model_errors=False, use_model=False, get_lander_state=None):
        self.zero_model_errors=zero_model_errors
        self.use_model = use_model
        self.get_lander_state = get_lander_state

    def run_episode(self, env, policy, val_func, model, recurrent_steps, use_timestep, add_padding=True):
        if self.get_lander_state is None:
            self.get_lander_state = env.lander.get_state_agent5
    
        obs = env.reset()
        muxed_observes , observes, actions,  rewards1, rewards2, nobserves,   model_states, model_errors, model_predicts,  policy_states,  vpreds   =      [], [], [], [], [], [], [], [], [], [], []
        traj = {}
        done = False
        step = 0.0
        model_state = model.get_prior(obs.astype(np.float64).reshape((1, -1)))
        model_error = model.net.initial_error
        model_predict, model_vpred  = model.get_initial_predict(obs.astype(np.float64).reshape((1, -1)))
        pv_state = self.get_lander_state(0.0).astype(np.float64).reshape((1, -1))
        policy_state = policy.net.initial_state

        flag = 1
        while not done:
            obs = obs.astype(np.float64).reshape((1, -1))
            model_states.append(model_state)
            model_errors.append(model_error)
            policy_states.append(policy_state)

            model_predicts.append(model_predict)
            vpreds.append(model_vpred)
 
            if use_timestep:
                obs = np.append(obs, [[step]], axis=1)  # add time step feature
            observes.append(obs)

            if self.use_model:
                obs_muxed = model_predict
            else:
                obs_muxed = pv_state
 
            muxed_observes.append(obs_muxed)
            action, env_action, policy_state = policy.sample(obs_muxed, policy_state)
            actions.append(action)
 
            obs, reward, done, reward_info = env.step(env_action)
            nobserves.append(obs.astype(np.float64).reshape((1, -1)))
           
            # test mode to test future predictions w/o feedback
            zeroed_obs = observes[-1]
            if self.zero_model_errors:
                model_error = np.zeros_like(model_error)
                if not flag:
                    zeroed_obs = model_predict
            print(flag)
            if flag:
                foo = observes[-1] 
            else:
                foo = model_predict
            model_predict,  model_vpred, model_state, model_error = model.predict(foo, action,   nobserves[-1], model_state, model_error, np.asarray([1]), np.asarray([flag]))

            #model_error = np.zeros_like(model_error)
            #model_predict,  model_vpred, model_state, model_error = model.predict(observes[-1], action,   nobserves[-1], model_state, model_error, np.asarray([1]), np.asarray([flag]))
 
            env.lander.trajectory['value'].append(model_vpred.copy())

            pv_state = self.get_lander_state(0.0).astype(np.float64).reshape((1, -1)) 
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

        env.lander.trajectory['value'].append(model_vpred.copy())

        traj['muxed_observes'] = np.concatenate(muxed_observes)
        traj['observes'] = np.concatenate(observes)
        traj['actions'] = np.concatenate(actions)
        traj['rewards1'] = np.array(rewards1, dtype=np.float64)
        traj['rewards2'] = np.array(rewards2, dtype=np.float64)
        traj['nobserves'] = np.concatenate(nobserves)
        traj['model_states'] = np.concatenate(model_states)
        traj['model_errors'] = np.concatenate(model_errors)
        traj['model_predicts'] = np.concatenate(model_predicts)
        traj['policy_states'] = np.concatenate(policy_states)
        traj['vpreds'] = np.array(vpreds, dtype=np.float64)

        traj['flags'] = np.zeros(len(observes))
        traj['flags'][0] = 1

        traj['masks'] = np.ones_like(traj['rewards1'])

        return traj

    def update_scalers(self, policy, val_func, model, rollouts):
        model.update_scalers(rollouts)
        policy.update_scalers(rollouts)

    def update(self,policy,val_func,model,rollouts, logger):
        model.fit(rollouts , logger)
        policy.update(rollouts, logger)

