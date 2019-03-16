import numpy as np

class Lidar_measurement_model(object):

    """
    position should be in asteroid centered reference frame
    Asteroid represented as a point cloud of vertices, but cut in half to avoid spurious readings

    """
    def __init__(self, shape_model, max_range=10000):
        self.shape_model = shape_model
        self.max_range = max_range
        print('Half Shape Model: ', np.min(shape_model,axis=0), np.max(shape_model,axis=0), shape_model.shape[0])
        m = shape_model.shape[0]
        D = []
        for i in range(m):
            v = shape_model[i]
            sm_f = shape_model[np.arange(m)!=i] 
            d = np.min(np.linalg.norm(sm_f-v,axis=1))
            D.append(d)
        print('Shape Model Vertex Spacing (mean / min / max): ', np.mean(D),np.min(D), np.max(D))
        self.miss_threshold = 2 *  np.max(D)

    # asteroid body-centered frame
 
    def get_range(self, pos, vel, dvec):
        test_dot = np.dot(-pos/np.linalg.norm(pos) , dvec)
        if test_dot < 0:
            return self.max_range, 0.0, None

        closest, dist = self.closest_vertex(self.shape_model, pos, dvec)
        if dist > self.miss_threshold:
            return self.max_range, 0.0, None

        range = np.linalg.norm(closest - pos)
        dopplar = -pos.dot(vel)/np.linalg.norm(pos-closest)
        #print('*closest: ', closest)
        #print('pos/vel: ', pos,vel)
        #print('range: ', range)
        #print('dopplar: ', dopplar)
        return range, dopplar, closest

    def ortho_proj(self, V,p,d):
        # vertices, position, dvec
        Q=V-p
        qdot = Q.dot(d)
        proj = (qdot * np.expand_dims(d, axis=1)).T
        oproj = Q-proj
        return oproj

    def closest_vertex(self, V,p,d):
        dist = np.linalg.norm(self.ortho_proj(V,p,d),axis=1)
        idx = np.argmin(dist)
        return V[idx], dist[idx]
