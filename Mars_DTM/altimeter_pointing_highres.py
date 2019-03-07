

import numpy as np
import env_utils as envu
import attitude_utils as attu

class Altimeter(object):
    def __init__(self, measurement_model, target_position, pointing_target=None, dphi=0.0,  theta1=np.pi/8, theta2=np.pi/16,  nbeams=4,  attitude_parameterization=attu.Quaternion_attitude()):
        self.measurement_model = measurement_model
        self.target_position = target_position
        self.theta1 = theta1
        self.theta2 = theta2

        self.nbeams = nbeams
        self.attitude_parameterization = attitude_parameterization
        self.refdvec_b = np.asarray([0.0, 0.0, -1.0])
        self.phis1 = np.linspace(-np.pi,np.pi-2*np.pi/nbeams,nbeams)
        self.phis2 = self.phis1.copy() + np.pi/4
        
        self.dphi = dphi
        if pointing_target is None:
            self.pointing_target = np.zeros(3)
        else:
            self.pointing_target = pointing_target
 
        self.dvecs_b = self.get_body_frame_attitude()
 
       
    def get_body_frame_attitude(self):
        #self.phis += self.dphi
        #self.phis = attu.vec_picheck(self.phis)
        dvecs_b1 = []
        dvecs_b2 = []
        for i in range(self.phis1.shape[0]):
            dvecs_b1.append(-np.asarray([np.sin(self.theta1)*np.cos(self.phis1[i]),
                                     np.sin(self.theta1)*np.sin(self.phis1[i]),
                                     np.cos(self.theta1)]))
        for i in range(self.phis2.shape[0]):
            dvecs_b2.append(-np.asarray([np.sin(self.theta2)*np.cos(self.phis2[i]),
                                     np.sin(self.theta2)*np.sin(self.phis2[i]),
                                     np.cos(self.theta2)]))

        dvecs_b2.append(np.asarray([0.,0.,-1.]))

        dvecs_b = np.vstack((dvecs_b1, dvecs_b2))
        return np.asarray(dvecs_b)
 

    def get_inertial_dvecs(self, dvecs_b, position, velocity):
        position1 = position - self.pointing_target
        dvec_i = -position1 / np.linalg.norm(position1) 
        C = attu.DCM(self.refdvec_b, dvec_i)
        dvecs_i = C.dot(dvecs_b.T).T
        return dvecs_i

        return dvecs_i

    def get_reading(self, position, velocity):
        dtm_position = position + self.target_position

        dvecs_b = self.get_body_frame_attitude()
        dvecs_i = self.get_inertial_dvecs(dvecs_b, position, velocity)
        altitudes = [] 
        cvs = []
        #print('FOO: ',dvecs_i)
        for dv in dvecs_i: 
            a, v, _ = self.measurement_model.get_altimeter_reading(dv, dtm_position, velocity)
            altitudes.append(a)  # was a.copy()
            cvs.append(v)  #.copy())
            #print('BAR: ', a, v, dv , velocity)
        #print(np.asarray(altitudes), np.asarray(cvs))
        return np.asarray(altitudes), np.asarray(cvs)
         
        
        
