import  numpy as np
from time import time

class DTM_measurement_model(object):

    def __init__(self,dtm,vertical_resolution=1.0,check_vertical_errors=False,max_reading=4000,print_missing_beams=False,print_stats_every=100000):
        self.dtm = dtm
        self.print_missing_beams = print_missing_beams
        self.vertical_resolution = vertical_resolution
        self.check_vertical_errors = check_vertical_errors
        self.print_stats_every=print_stats_every
        self.min_elevation = np.min(dtm)
        self.max_elevation = np.max(dtm)
        self.dtm_x = dtm.shape[0]
        self.dtm_y = dtm.shape[1]
        self.dtm_normal = 1.0*np.asarray([0,0,1])
        self.dtm_reference = np.zeros(3)
        self.nref = int(np.round(self.max_elevation - self.min_elevation))+1
        self.middle = self.nref//2
        self.dtm_reference_array = np.zeros((self.nref,3))
        for i in range(self.nref):
            self.dtm_reference_array[i,2] = vertical_resolution * i
        self.max_reading = max_reading
        self.miss_cnt = 0
        self.reading_cnt = 0
        print('DTM MM: nref fixed: ',self.nref, self.dtm_x, self.dtm_y)

    def lp_intersect(self,plane_normal, plane_point, ray_dir, ray_point, epsilon=1e-6):
        ndotu = plane_normal.dot(ray_dir)

        if np.abs(np.dot(ray_dir,plane_normal))<epsilon:
            return None
        w = ray_point - plane_point
        si = -plane_normal.dot(w) / plane_normal.dot(ray_dir)

        intersect = ray_point + si * ray_dir
        return intersect

    def lp_intersect_vectorized(self,plane_normal, plane_point_list, ray_dir, ray_point, epsilon=1e-6):
        ndotu = plane_normal.dot(ray_dir)

        if np.abs(np.dot(ray_dir,plane_normal))<epsilon:
            return None
        W = ray_point - plane_point_list
        si = -W.dot(plane_normal) / plane_normal.dot(ray_dir)
        si = np.expand_dims(si,axis=1)
        intersect = ray_point + si * ray_dir
        return intersect

    def get_altimeter_reading(self, altimeter_dvec, altimeter_location, altimeter_velocity, debug=False):
        # get x,y components of intersection to index map
        intersections = self.lp_intersect_vectorized(self.dtm_normal, self.dtm_reference_array, altimeter_dvec, altimeter_location)
        #assert intersections is not None
        if intersections is None:
            if self.print_missing_beams:
                print('Altimeter Beam Missing All Planes')
            self.miss_cnt += 1
            return self.max_reading, 0.0 , np.zeros(3)

        intersection_elevations = intersections[:,2]
        intersection_coords = np.round(intersections[:,0:2]).astype(int)
        if debug:
            print('intersection_elevations: ',intersection_elevations)
            print('intersections: ', intersections)
            print('intersection_coords: ', intersection_coords)
        # check that all intersections are w/in dtm range
        bad1 =   np.any(intersection_coords >= np.asarray([self.dtm_x,self.dtm_y]))
        bad2 =   np.any(intersection_coords <= np.asarray([0,0]))
        if bad1 or bad2:
            if self.print_missing_beams:
                print('Altimeter Beam Missing Map:')
                bad_idx1 = np.where(intersection_coords >= np.asarray([self.dtm_x,self.dtm_y]))[0]
                bad_idx2 = np.where(intersection_coords <= np.asarray([0,0]))[0]
                if bad1:
                    print('\tIntersection Coords: ', intersection_coords[bad_idx1[0]])
                if bad2:
                    print('\tIntersection Coords: ', intersection_coords[bad_idx2[0]])
                print('\tAltimeter Location: ', altimeter_location) 
                print('\tAltimeter Dvec: ', altimeter_dvec)
            self.miss_cnt += 1
            return self.max_reading, 0.0 , np.zeros(3)
    
        dtm_elevations = self.dtm[intersection_coords[:,0],intersection_coords[:,1]] 
        
        deltas = np.abs(intersection_elevations - dtm_elevations)

        closest = np.argmin(deltas)
        intercept = intersections[closest]
        # pick closest elevation 
        altitude = np.linalg.norm(intercept - altimeter_location)
        x = int(intercept[0])
        y = int(intercept[1])
        if self.check_vertical_errors:
            error = altitude-(altimeter_location[2]-self.dtm[x,y])
        if self.check_vertical_errors and np.abs(error) > 10:
            print('DEBUG: ',altitude,altimeter_location[2]-self.dtm[x,y])
            print('ERROR: ',error)
            print(np.min(deltas), np.argmin(deltas), altimeter_location, altitude, altimeter_location[2]-self.dtm[x,y])
            for i in range(intersection_elevations.shape[0]):
                print(i, intersection_elevations[i], dtm_elevations[i])
            print('IC: ',np.min(intersection_coords,axis=0),np.max(intersection_coords,axis=0))
            assert False


        v_doppler = self.get_vc(altimeter_dvec,altimeter_velocity)
        #print(altitude, altimeter_dvec,altimeter_velocity,v_doppler)
        self.reading_cnt += 1
        if self.reading_cnt % self.print_stats_every == 0:
            print('DTM Model Miss Ratio: ', self.miss_cnt / self.reading_cnt, self.miss_cnt)
        return altitude, v_doppler, intercept 
 
    def get_vc(self,r_tm, v_tm):
        vc = -r_tm.dot(v_tm)/np.linalg.norm(r_tm)
        return vc
 
