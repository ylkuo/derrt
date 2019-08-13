import numpy as np

def dist(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    return np.linalg.norm(v1 - v2)

def unit_vector(vector):
    if np.linalg.norm(vector) > 0:
        return vector / np.linalg.norm(vector)
    else:
        return vector

def theta_from_vecs(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    theta = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if np.cross(v1, v2) < 0 and theta > 0:
        theta *= -1
    return theta
