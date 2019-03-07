

import numpy as np
import env_utils as envu
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
 
        self.dvecs_b = self.get_body_frame_attitude()
 
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
        dvec_0 = np.asarray([0.0,0.0,-1.0])
        dvec_i /= np.linalg.norm(dvec_i)
        C = attu.DCM(self.refdvec_b, dvec_i)
        dvecs_i = C.dot(dvecs_b.T).T

        return dvecs_i

    def get_reading(self, position, velocity):
        dtm_position = position + self.target_position
        dtm_position_list = []
        dtm_position_list.append(dtm_position + np.asarray([ 100., -100., 0.]) )
        dtm_position_list.append(dtm_position + np.asarray([ 100.,  100., 0.]) )
        dtm_position_list.append(dtm_position + np.asarray([-100., -100., 0.]) )
        dtm_position_list.append(dtm_position + np.asarray([-100.,  100., 0.]) )


        altitudes = [] 
        cvs = []
        dv =  np.asarray([0.0,0.0,-1.0])
        for pos in dtm_position_list: 
            a, v, _ = self.measurement_model.get_altimeter_reading(dv, pos, velocity)
            altitudes.append(a)  # was a.copy()
            cvs.append(v)  #.copy())
        return np.asarray(altitudes), np.asarray(cvs)
         
        
        
