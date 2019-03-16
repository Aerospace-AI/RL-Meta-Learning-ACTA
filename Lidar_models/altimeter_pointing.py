

import numpy as np
import attitude_utils as attu

class Altimeter(object):
    def __init__(self, measurement_model, target_position,  dphi=0.0, theta=np.pi/8, nbeams=4,  attitude_parameterization=attu.Quaternion_attitude()):
        self.measurement_model = measurement_model
        self.target_position = target_position
        self.theta = theta
        self.nbeams = nbeams
        self.attitude_parameterization = attitude_parameterization
        self.refdvec_b = np.asarray([0.0, 0.0, -1.0])
        self.phis = np.linspace(-np.pi,np.pi-2*np.pi/nbeams,nbeams) 
        self.dphi = dphi
        self.miss_cnt = 0
        self.cnt = 0
 
        self.dvecs_b = self.get_body_frame_attitude()
        #print(self.dvecs_b)
 
    def get_body_frame_attitude(self):
        self.phis += self.dphi 
        self.phis = attu.vec_picheck(self.phis)
        dvecs_b = []
        for i in range(self.phis.shape[0]):
            dvecs_b.append(-np.asarray([np.sin(self.theta)*np.cos(self.phis[i]),
                                     np.sin(self.theta)*np.sin(self.phis[i]),
                                     np.cos(self.theta)]))
        
        return  np.asarray(dvecs_b)
        

    def get_inertial_dvecs(self, dvecs_b, position, velocity):
        dvec_1 = velocity / (np.linalg.norm(velocity) + 1e-8)
        dvec_i = -position / np.linalg.norm(position) 
        dvec_i /= np.linalg.norm(dvec_i)
        C = attu.DCM3(self.refdvec_b, dvec_i)
        dvecs_i = C.dot(dvecs_b.T).T

        return dvecs_i

    def get_reading(self, position, velocity):
        dtm_position = position + self.target_position

        dvecs_i = self.get_inertial_dvecs(self.dvecs_b, position, velocity)
        altitudes = [] 
        cvs = []
        vtests = []

        for dv in dvecs_i:
            #print('DV: ', dv) 
            a, v, v_test = self.measurement_model.get_range(dtm_position, velocity, dv)
            altitudes.append(a)  # was a.copy()
            cvs.append(v) 
            vtests.append(v_test)
        miss = np.all(v_test is None)
        if miss:
            self.miss_cnt += 1
        self.cnt += 1
        if self.cnt % 10000 == 0:
            print('Miss Ratio: ', self.miss_cnt / self.cnt)
 
        return np.asarray(altitudes), np.asarray(cvs)
         
        
        
