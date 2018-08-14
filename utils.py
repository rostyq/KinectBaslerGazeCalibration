import numpy as np

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