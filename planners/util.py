import numpy as np
import torchvision

epsilon = 1e-8

def argmin(fn, sequence):
    values = list(sequence)
    scores = [fn(x) for x in values]
    return values[scores.index(min(scores))]

def argmax(fn, sequence):
    values = list(sequence)
    scores = [fn(x) for x in values]
    return values[scores.index(max(scores))]

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

def make_filter_image(layer, use_color=True, scale_each=True):
    """Build an image of the weights of the filters in a given convolutional layer."""
    weights = layer.weight.data.to("cpu")
    if not use_color:
        n_input_channels = weights.size()[1]
        weights = weights.view([weights.size()[0], 1, weights.size()[1]*weights.size()[2], weights.size()[3]])
    img = torchvision.utils.make_grid(weights, normalize=True, scale_each=scale_each)
    return img
