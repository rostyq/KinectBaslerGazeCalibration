class PersonCalibration:
    
    default_parameters = np.array([0.0, 0.0, avg_eye_radius, avg_eye_radius/2, -0.03,  0.03, 0.05])
    
    def __init__(self, name='person', parameters=None):
        
        self.name = name
        
        # person parameters
        self.parameters = np.array(parameters) if parameters else self.default_parameters

    @property
    def parameters(self):
        self._parameters = None
    
    @parameters.getter
    def parameters(self):
        return self._parameters
    
    @parameters.setter
    def parameters(self, value):
        self._parameters = value
        
#         print(f'Set parameters to: {value}')
        
        # person parameters links
        self.visual_angles = self.parameters[:2].reshape(1, 2)
        self.eye_radius = self.parameters[2]
        self.cornea_radius = self.parameters[3]
        self.eye_center_head = self.parameters[4:7].reshape(1, 3)
        
    def calculate_eye_center(self, rotation, translation):
        """
        rotation (1, 3)
        translation (1, 3)

        parameters
        eye_center_head (1, 3)
        """
        # (Rodrigues((1, 3)) -> (3, 3))
        rotation_matrix = cv2.Rodrigues(rotation)[0]

        # ((3, 3) @ (3, 1) + (1, 3).T).T -> (1, 3)
        return (rotation_matrix @ self.eye_center_head.T + translation.T).T
    
    def calculate_optical_axis(self, pupil_center, eye_center):
        # ((1, 3) - (1, 3)) / scalar -> optical axis (1, 3)
        return (pupil_center - eye_center) / np.linalg.norm(pupil_center - eye_center)
    
    def calculate_cornea_center(self, eye_center, optical_axis):
        # (1, 3) + scalar * (1, 3) -> (1, 3)
        return eye_center + self.cornea_radius * optical_axis
    
    def calculate_visual_axis(self, optical_axis):
        # (((1, 3) -> (1, 2)) + (1, 2)) -> (1, 3)
        visual_axis = angles2vecs(vecs2angles(optical_axis) + self.visual_angles)
        
        # norm vector
        return visual_axis / np.linalg.norm(visual_axis)
    
    def calculate_eye_geometry(self, pupil_center, rotation, translation):
        
        eye_center = self.calculate_eye_center(rotation, translation)
        optical_axis = self.calculate_optical_axis(pupil_center, eye_center)
        cornea_center = self.calculate_cornea_center(eye_center, optical_axis)
        visual_axis = self.calculate_visual_axis(optical_axis)
        
        return cornea_center, visual_axis
    
    def gaze_direction(self, pupil_center, rotation, translation, coeff=3):
        """
        pupil_center (1, 3)
        rotation (1, 3)
        translation (1, 3)

        parameters
        eye_center_head (1, 3)
        cornea_radius scalar
        """
        cornea_center, visual_axis = self.calculate_eye_geometry(pupil_center, rotation, translation)
        
        return np.array([cornea_center.flatten(), (cornea_center + coeff * visual_axis).flatten()])
    
    def calibrate(self, points, pupil_centers, rotations, translations, alpha=1e-2):
        
        arr = np.column_stack((points, pupil_centers, rotations, translations))

        def cost(parameters):
            
            self.parameters = parameters
            
            loss = 0
            
            for s in arr:
                
                point = s[:3].reshape(1, 3)
                pupil_center, rotation, translation = s[3:6].reshape(1, 3), s[6:9].reshape(1, 3), s[9:12].reshape(1, 3)
                
                cornea_center, visual_axis = self.calculate_eye_geometry(pupil_center, rotation, translation)
                
                gaze_vector_true = point - cornea_center
                gaze_vector_true = gaze_vector_true / np.linalg.norm(gaze_vector_true)
                
                loss += np.sum((gaze_vector_true - visual_axis) ** 2) + alpha * np.linalg.norm(self.eye_center_head) ** 2
            
            return loss
        
        constraints = [
            # eye radius
            {'type': 'ineq', 'fun': lambda x: avg_eye_radius * 0.7 - x[2]},
            {'type': 'ineq', 'fun': lambda x: x[2] - avg_eye_radius * 1.3},
            
            # cornea radius
            {'type': 'ineq', 'fun': lambda x: x[2] - x[3]},
            {'type': 'ineq', 'fun': lambda x: x[3] - x[2] / 4},
            
            # visual axis angles
            {'type': 'ineq', 'fun': lambda x: x[0] - np.deg2rad(5)},
            {'type': 'ineq', 'fun': lambda x: np.deg2rad(5) - x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1] - np.deg2rad(5)},
            {'type': 'ineq', 'fun': lambda x: np.deg2rad(5) - x[1]},
            
            # eye center head
            {'type': 'ineq', 'fun': lambda x: abs(x[4]) - 0.05},
            {'type': 'ineq', 'fun': lambda x: x[5] - 0.05},
            {'type': 'ineq', 'fun': lambda x: x[6] - 0.07},
            {'type': 'ineq', 'fun': lambda x: 0.05 - abs(x[4])},
            {'type': 'ineq', 'fun': lambda x: 0.05 - x[5]},
            {'type': 'ineq', 'fun': lambda x: 0.07 - x[6]},
            
        ]
        minimized = minimize(cost, x0=self.parameters, constraints=constraints, method='SLSQP')
        
        return minimized
        