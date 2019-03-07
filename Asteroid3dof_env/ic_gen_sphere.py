import numpy as np
import attitude_utils as attu

class Landing_icgen(object):

    def __init__(self,
                 position_r=(900,1100) , position_theta=(0, np.pi/2), position_phi=(-np.pi,np.pi),
                 velocity_x=(-0.10,0.10), velocity_y=(-0.10,0.10), velocity_z=(-0.10,0.10),
                 scale=None, debug=False, 
                 noise_u=np.zeros(3), noise_sd=np.zeros(3), adjust_apf_v0=False,
                 M=(2e10,20e10), 
                 min_w=(-1e-3,-1e-3,-1e-3), max_w=(1e-3,1e-3,1e-3),
                 min_srp=(-1e-6,-1e-6,-1e-6), max_srp=(1e-6,1e-6,1e-6) ,  
                 min_mass=450, max_mass=500):

        self.position_r = position_r
        self.position_theta = position_theta
        self.position_phi = position_phi
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.velocity_z = velocity_z
 
        self.noise_u = noise_u
        self.noise_sd = noise_sd
        self.M     = M
        self.min_w = min_w
        self.max_w = max_w
        self.min_srp = min_srp
        self.max_srp = max_srp
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.debug = debug
        self.adjust_apf_v0 = adjust_apf_v0 


 

    def show(self):
        print('Landing_icgen:')
        print('    adjust_apf_v0               : ',self.adjust_apf_v0)
        print('    w (min/max)                 : ',self.min_w, self.max_w)
        print(' mass (min/max)                 : ',self.min_mass, self.max_mass)
        print('  srp (min/max)                 : ',self.min_srp, self.max_srp)
        print('M (min/max)                     : ',self.M)
        print('Position Theta                  : ',self.position_theta)
        print('Position Phi                    : ',self.position_phi)

    def set_ic(self , lander, dynamics):

        dynamics.noise_u = np.random.uniform(low=-self.noise_u, high=self.noise_u,size=3)
        dynamics.noise_sd = self.noise_sd

        dynamics.M =  np.random.uniform(low=self.M[0], high=self.M[1])

        dynamics.w =  np.random.uniform(low=self.min_w, high=self.max_w, size=3)

        dynamics.srp =  np.random.uniform(low=self.min_srp, high=self.max_srp, size=3)

        lander.init_mass = np.random.uniform(low=self.min_mass, high=self.max_mass)

        theta = np.random.uniform(low=self.position_theta[0],   high=self.position_theta[1])
        phi   = np.random.uniform(low=self.position_phi[0],     high=self.position_phi[1])
        r     = np.random.uniform(low=self.position_r[0],       high=self.position_r[1])

        rx = r * np.sin(theta) * np.cos(phi)
        ry = r * np.sin(theta) * np.sin(phi)
        rz = r * np.cos(theta)

        vx = np.random.uniform(low=self.velocity_x[0], high=self.velocity_x[1])
        vy = np.random.uniform(low=self.velocity_y[0], high=self.velocity_y[1])
        vz = np.random.uniform(low=self.velocity_z[0], high=self.velocity_z[1])

        if self.debug:
            print('g: ',dynamics.g)
            print('w: ',dynamics.w)
            print('srp: ',dynamics.srp)
            print('mass: ',lander.init_mass)
            print('rx,vx: ', rx,vx)
            print('ry,vy: ', ry,vy)
            print('rz,vz: ', rz,vz)

            
        lander.state['position'] = np.asarray([rx, ry, rz])
        lander.state['velocity'] = np.asarray([vx, vy, vz])
        lander.state['disturbance'] = np.zeros(3) 
        lander.state['thrust'] = np.ones(3)*lander.min_thrust 
        lander.state['mass']   = lander.init_mass

        lander.apf_state = False
        if self.adjust_apf_v0:
            lander.apf_v0 = np.linalg.norm(lander.state['velocity'])
