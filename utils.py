import numpy as np
import cv2


class Color:
    
    Blue = (0, 0, 255)
    Red = (255, 0, 0)
    Green = (0, 255, 0)
    Black = (0, 0, 0)
    White = (255, 255, 255)
    

class HDFace:
    
    All = list(range(0, 1347))
    
    LeftEyeInnercorner = 210
    LeftEyeOutercorner = 469
    LeftEyeMidtop = 241
    LeftEyeMidbottom = 1104
    
    RightEyeInnercorner = 843
    RightEyeOutercorner = 1117
    RightEyeMidtop = 731
    RightEyeMidbottom = 1090
    
    LeftEyeBrowInner = 346
    LeftEyeBrowOuter = 140
    LeftEyeBrowCenter = 222
    
    RightEyeBrowInner = 803
    RightEyeBrowOuter = 758
    RightEyeBrowCenter = 849
    
    RightEyeInnerLid = [776,  777,  846,  843,  1098, 
                         1097,  1095,  1096,  1096,  1091, 
                         1090,  1092,  1099,  1094,  1093, 
                         1100,  1101,  1102,  1117,  1071, 
                         1012,  992,  987,  752,  749, 
                         876,  733,  731,  728]
    LeftEyeInnerLid = [210, 316, 187, 153, 121,
                        241, 244, 238, 137, 211,
                        188, 287, 440, 1116, 469,
                        1115, 1114, 1113, 1107, 1106,
                        1112, 1105, 1104, 1103, 1108,
                        1109, 1111, 1110]
    
    MouthLeftcorner = 91
    MouthRightcorner = 687
    MouthUpperlipMidtop = 19
    MouthUpperlipMidbottom = 1072
    MouthLowerlipMidtop = 10
    MouthLowerlipMidbottom = 8
    
    NoseTip = 18
    NoseBottom = 14
    NoseBottomLeft = 156
    NoseBottomRight = 783
    NoseTop = 24
    NoseTopLeft = 151
    NoseTopRight = 772
    
    ForeheadCenter = 28
    
    LeftCheekCenter = 412
    RightCheekCenter = 933
    LeftCheekbone = 458
    RightCheekbone = 674
    
    ChinCenter = 4
    
    LowerJawLeftEnd = 1307
    LowerJawRightEnd = 1327
    
    @classmethod
    def items(cls):
        return {attr_name: getattr(cls, attr_name)
                for attr_name in dir(cls) if not attr_name.startswith('__') and not callable(getattr(cls, attr_name))}
    
    @classmethod
    def values(cls):
        return iter([value for key, value in cls.items().items()])
    
    @classmethod
    def keys(cls):
        return iter([key for key, value in cls.items().items()])
    

def vecs2angles(vectors):
    """
    theta = asin(-y) -- pitch
    phi = atan2(-x, -z) -- yaw
    """
    x, y, z = vectors.T

    pitch = np.arcsin(-y)
    yaw = np.arctan2(-x, -z)

    return np.column_stack((yaw, pitch))


def angles2vecs(angles):
    """
    x = (-1) * cos(pitch) * sin(yaw)
    y = (-1) * sin(pitch)
    z = (-1) * cos(pitch) * cos(yaw)
    """
    yaw, pitch = angles.T

    x = (-1) * np.cos(pitch) * np.sin(yaw)
    y = (-1) * np.sin(pitch)
    z = (-1) * np.cos(pitch) * np.cos(yaw)

    vectors = np.column_stack((x, y, z))
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / norm


def haversin(*args, **kwargs):
    return (1 - np.cos(*args, **kwargs)) / 2


def archaversin(*args, **kwargs):
    return np.arccos(1 - 2 * args[0], *args[1:], **kwargs)


def calc_angle_3d(v1, v2, deg=True):

    assert v1.ndim == 2 and v2.ndim == 2, 'Wrong `ndim`.'
    assert v1.shape[1] == 3 and v2.shape[1] == 3, 'Wrong `shape`.'

    norm_v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    norm_v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

    cos_angle = np.sum(norm_v1 * norm_v2, axis=1)
    angle = np.arccos(cos_angle)
    if deg:
        return np.rad2deg(angle)
    else:
        return angle


def calc_angle_spherical(a1, a2, deg=True):

    assert a1.ndim == 2 and a2.ndim == 2, 'Wrong `ndim`.'
    assert a1.shape[1] == 2 and a2.shape[1] == 2, 'Wrong `shape`.'

    yaw1, pitch1 = a1.T
    yaw2, pitch2 = a2.T

    product = np.cos(pitch1) * np.cos(pitch2)
    haversin_d_pitch = haversin(pitch2 - pitch1)
    haversin_d_yaw = haversin(yaw2 - yaw1)
    haversin_angle = haversin_d_pitch + product * haversin_d_yaw
    
    angle = archaversin(haversin_angle)
    if deg:
        return np.rad2deg(angle)
    else:
        return angle
    
    
def to_tuple(dct):
    return tuple(int(item) for key, item in dct.items())


def extract_rectangle(img, pt0, pt1):
    return img[pt0[1]:pt1[1], pt0[0]:pt1[0]]


def dict_to_np(dict_data):
    return np.array([value for axis, value in dict_data.items()]).reshape(-1, len(dict_data))


def add_dummy_z(arr_2d):
    if isinstance(arr_2d, dict):
        arr_2d = dict_to_np(arr_2d)
    return np.column_stack((arr_2d, np.full(arr_2d.shape[0], 1)))


def to_json(vector, scale=1):
    return {axis: value * scale for axis, value in zip('XYZ', vector)}


def vectors_to(vectors, rotations, translations=None, to='self'):
    if translations is None:
        translations = np.zeros(vectors.shape)
        
    arr = zip(vectors, rotations, translations)

    if to is 'self':
        return np.array([(np.linalg.inv(cv2.Rodrigues(R)[0]) @ (V - T).T).flatten() for V, R, T in arr])
    elif to is 'origin':
        return np.array([(cv2.Rodrigues(R)[0] @ V.T + T.T).flatten() for V, R, T in arr])
    
    
def quaternion_to_rotation(quaternion):
    """
    Converts angle-axis to quaternion
    :param quaternion: dict {'X': , 'Y': , 'Z': , 'W': }
    :return: angle-axis rotation vector
    """
    if isinstance(quaternion, dict):
        quaternion = dict_to_np(quaternion).flatten()
        
    assert quaternion.ndim == 1
    assert quaternion.shape == (4,)
    
    t = np.sqrt(1 - quaternion[-1] ** 2)
    
    if t:
        return (quaternion[:3] / t)
    else:
        return np.zeros((3,))
    
    
def plane_line_intersection(line_points, plane_points):

    """
    Compute intersection point of plane and lineself.
    Parameter line_points consists of two points and stands to determine
    line's equesion:
        (x - x_1)/(x_2 - x_1) =
       =(y - y_1)/(y_2 - y_1) =
       =(z - z_1)/(z_2 - z_1).
    Parameter plane_points consists of three points and stands to determine
    plane's equation:
        A*x + B*y + C*z = D.
    This function returns 3D coordinates of intersection point.
    """
    
    assert isinstance(line_points, np.ndarray), str(type(line_points))
    assert isinstance(plane_points, np.ndarray), str(type(plane_points))
    
    assert line_points.shape == (2, 3), str(line_points.shape)
    assert plane_points.shape == (3, 3), str(plane_points.shape) 
    
    line_point_1, line_point_2 = line_points
    plane_point_1, plane_point_2, plane_point_3 = plane_points

    # These two vectors are in the plane.
    vector_1 = plane_point_3 - plane_point_1
    vector_2 = plane_point_2 - plane_point_1

    # The cross prodaction is a normal vector to the plane.
    A3 = np.cross(vector_1, vector_2)
    B3 = np.dot(A3, plane_point_3)

    # Compute the solution of equasion A * x = B.
    # Compute matrix A.
    line_points_diff = line_point_2 - line_point_1
    A1 = np.array([1, -1]) / line_points_diff[:2] 
    A2 = np.array([1, -1]) / line_points_diff[1:]
    
    A = np.array([[*A1, 0.0], [0.0, *A2], [*A3]])

    # Compute vector B.
    B1 = np.sum(line_point_1[:2] * A1)
    B2 = np.sum(line_point_1[1:] * A2)
    
    B = np.array([B1, B2, B3])
    
    # Compute intersection point.
    return np.linalg.solve(A, B)
