import env_utils as envu
import numpy as np
from time import time

class Dynamics_model(object):

    """
        The dynamics model take a lander model object (and later an obstacle object) and modifies  
        the state of the lander.

        The lander object instantiates an engine model, that maps body frame thrust and torque to
        the inertial frame.  Note that each lander can have its own intertial frame which can be 
        centered on the lander's target. 

        Currentlly this model does not model environmental dynamics, will be added later
 
        The lander model maintains a state vector: 
            position                                [0:3]
            velocity                                [3:6]
            mass                                    [7]     
                 

    """

    def __init__(self, h=0.5, noise_u=np.zeros(3),noise_sd=np.zeros(3), w=np.zeros(3), g=np.zeros(3), srp=np.zeros(3), landing_target=np.asarray([250., 0., 0.])):
        self.h = h
        self.landing_target = landing_target
        self.Isp = 210.0
        self.g_o = 9.81
        self.w = w
        self.g = g 
        self.noise_sd = noise_sd 
        self.noise_u =  noise_u
        self.srp = srp
        print('3-dof dynamics model')

    def next(self,t,thrust_cmd,lander):
        t0 = time()
        #lander.prev_state = lander.state.copy()
        pos = lander.state['position']
        vel = lander.state['velocity']

        # centrifugal force requires spaacecraft position in asteroid centered frame 
        ac_pos = pos + self.landing_target

        coriolis_acc = np.cross(2 * vel , self.w )
        centrifugal_acc =  np.cross(np.cross(self.w, ac_pos), self.w)
        env_acc = self.g + coriolis_acc + centrifugal_acc + self.srp
        #print('env_force: ',env_acc * lander.state['mass'], lander.state['mass'])
        thrust = envu.limit_thrust(thrust_cmd, lander.min_thrust, lander.max_thrust)
        noise = (self.noise_u + self.noise_sd * np.random.normal(size=3)) /  lander.state['mass']
        #
        # Use 4th order Runge Kutta to integrate equations of motion
        #

        x = lander.get_state_dynamics()
        ode = lambda t,x : self.eqom(t, x, thrust, env_acc, noise)
        x_next = envu.rk4(t, x, ode, self.h )

        lander.state['position'] = x_next[0:3]
        lander.state['velocity'] = x_next[3:6]
        lander.state['mass']     = np.maximum(x_next[6], lander.dry_mass)
        lander.state['thrust'] = thrust 
        return x_next

    
     
           
    def eqom(self,t, x, thrust, env_acc, noise):

        r = x[0:3]
        v = x[3:6]
        m = x[6]
        xdot = np.zeros(7)
        xdot[0:3] = v
        xdot[3:6] = thrust / m + env_acc  + noise
        xdot[6] = -np.linalg.norm(thrust) / (self.Isp * self.g_o) 

        return xdot
 
       
        
        
