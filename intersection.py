
from scipy.optimize import minimize


class Intersection:
    
    def __init__(self, cam_point, eye_radius, alpha=0.0001, tol=None):
        
        # model parameters
        self.eye_radius = eye_radius
        self.alpha = alpha
        self.tol = tol
        
        # points in 3d
        self.cam_point = cam_point
        self.eye_center = None
        self.ray_point = None
        
        
    def cost(self, point):
        
        # point must be on sphere surface
        sphere_cost = np.abs(np.sum((point - self.eye_center) ** 2) - (self.eye_radius ** 2))
        
        # point must be on line
        line_eq_components = (point - self.cam_point) / (self.ray_point - self.cam_point)
#         print(sphere_cost)
        line_cost = np.sum(np.abs(np.diff((point - self.cam_point) / (self.ray_point - self.cam_point))))
        
        # to prevent 2 point intersection ambiguity
        regularization = np.linalg.norm(point)
        
        # sum of all costs
        return sphere_cost ** 2 + line_cost ** 2 + sphere_cost * line_cost + self.alpha * regularization
    
    def find(self, eye_center, ray_point, verbose=True):
        
        # assign points
        self.eye_center = eye_center
        self.ray_point = ray_point
        
        # try minimize
        result = minimize(self.cost, x0=self.eye_center - self.ray_point, tol=self.tol)
        
        # print message
        if verbose:
            print(f'Cost: {result.fun:.2e}. ', result.message, sep='')
            
        return result.x
    
    def np_find(self, arr):
        return self.find(eye_center=arr[:3], ray_point=arr[3:], verbose=False)