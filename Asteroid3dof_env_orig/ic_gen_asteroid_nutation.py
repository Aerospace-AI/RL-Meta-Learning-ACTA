import numpy as np
import attitude_utils as attu

class Landing_icgen(object):

    def __init__(self,  downrange=(-100,100 , -0.5, 0.5), crossrange=(-100,100 , -0.5, 0.5),  altitude=(1400,1500,-1.0, -0.9) , 
                 scale=None, debug=False, 
                 noise_u=np.zeros(3), noise_sd=np.zeros(3), adjust_apf_v0=False, 
                 min_g=(1e-6,1e-6,1e-6),  max_g=(100e-6,100e-6,100e-6), 
                 min_w=(-1e-3,-1e-3,-1e-3), max_w=(1e-3,1e-3,1e-3),
                 min_srp=(-1e-6,-1e-6,-1e-6), max_srp=(1e-6,1e-6,1e-6) ,
                 w_o=2*np.pi/400,  
                 min_mass=450, max_mass=500): 
        self.downrange = downrange
        self.crossrange = crossrange
        self.altitude = altitude
        self.noise_u = noise_u
        self.noise_sd = noise_sd
        self.min_g = min_g
        self.max_g = max_g
        self.min_w = min_w
        self.max_w = max_w
        self.min_srp = min_srp
        self.max_srp = max_srp
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.debug = debug
        self.adjust_apf_v0 = adjust_apf_v0 

        assert downrange[0] <= downrange[1]
        assert downrange[2] <= downrange[3]
        assert crossrange[0] <= crossrange[1]
        assert crossrange[2] <= crossrange[3]
        assert altitude[0] <= altitude[1]
        assert altitude[2] <= altitude[3]

        if scale is not None:
            self.downrange = tuple([x / scale for x in downrange])
            self.crossrange = tuple([x / scale for x in crossrange])
            self.altitude = tuple([x / scale for x in altitude])
 

    def show(self):
        print('Landing_icgen:')
        print('    downrange                   : ',self.downrange)
        print('    crossrange                  : ',self.crossrange)
        print('    altitude                    : ',self.altitude)
        print('    adjust_apf_v0               : ',self.adjust_apf_v0)
        print('    g (min/max)                 : ',self.min_g, self.max_g)
        print('    w (min/max)                 : ',self.min_w, self.max_w)
        print(' mass (min/max)                 : ',self.min_mass, self.max_mass)
        print('  srp (min/max)                 : ',self.min_srp, self.max_srp)
 
    def set_ic(self , lander, dynamics):

        dynamics.noise_u = np.random.uniform(low=-self.noise_u, high=self.noise_u,size=3)
        dynamics.noise_sd = self.noise_sd

        dynamics.g =  np.random.uniform(low=self.min_g, high=self.max_g, size=3)

        dynamics.w =  np.random.uniform(low=self.min_w, high=self.max_w, size=3)

        dynamics.srp =  np.random.uniform(low=self.min_srp, high=self.max_srp, size=3)

        lander.init_mass = np.random.uniform(low=self.min_mass, high=self.max_mass)

        if self.debug:
            print('g: ',dynamics.g)
            print('w: ',dynamics.w)
            print('srp: ',dynamics.srp)
            print('mass: ',lander.init_mass)                                       
     
        r_downrange = np.random.uniform(low=self.downrange[0], high=self.downrange[1])
        r_crossrange = np.random.uniform(low=self.crossrange[0], high=self.crossrange[1])
        r_altitude = np.random.uniform(low=self.altitude[0], high=self.altitude[1])

        v_downrange = np.random.uniform(low=self.downrange[2], high=self.downrange[3])
        v_crossrange = np.random.uniform(low=self.crossrange[2], high=self.crossrange[3])
        v_altitude = np.random.uniform(low=self.altitude[2], high=self.altitude[3])

        lander.state['position'] = np.asarray([r_downrange,r_crossrange,r_altitude])
        lander.state['velocity'] = np.asarray([v_downrange,v_crossrange,v_altitude])
        lander.state['thrust'] = np.ones(3)*lander.min_thrust 
        lander.state['mass']   = lander.init_mass

        lander.apf_state = False
        if self.adjust_apf_v0:
            lander.apf_v0 = np.linalg.norm(lander.state['velocity'])
