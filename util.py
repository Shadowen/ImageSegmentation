import numpy as np


def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1, a2, b1, b2):
    if np.all(a1 == b1) or np.all(a1 == b2):
        return True, a1
    if np.all(a2 == b1) or np.all(a2 == b2):
        return True, a2

    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    if denom == 0:
        return False, None

    num = np.dot(dap, dp)
    intersection = (num / denom.astype(float)) * db + b1
    does_intersect = min(a1[0], a2[0]) <= intersection[0] <= max(a1[0], a2[0]) and min(b1[0], b2[0]) <= intersection[
        0] <= max(b1[0], b2[0])
    return does_intersect, intersection


def iterate_in_ntuples(lst, n, offset=0, loop=True):
    """
    Iterates through a list in overlapping tuples of n starting from the given offset.
    If loop=True, then the tuple (last, first) will be generated. Otherwise, it will not.
    """
    for i in range(len(lst) if loop else (len(lst) - n + 1)):
        yield [lst[(i + e + offset) % len(lst)] for e in range(n)]


def create_history(vertices, end_idx, history_length, image_size):
    """ Creates `history_length` frames, ending with `vertices[end_idx]` """
    num_vertices = len(vertices)
    history_mask = np.zeros([image_size, image_size, history_length])
    for i in range(history_length):
        x, y = vertices[(end_idx - history_length + i + 1) % num_vertices]
        history_mask[y, x, i] = 1
    return history_mask


def create_point_mask(point, size):
    mask = np.zeros([size, size])
    mask[point[1], point[0]] = 1
    return mask


def create_shape_mask(vertices, image_size):
    import skimage.draw
    mask = np.zeros([image_size, image_size])
    rr, cc = skimage.draw.polygon(vertices[:, 1], vertices[:, 0])
    mask[rr, cc] = 1
    return mask


class lazyproperty(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable datasets, as it replaces itself.
    '''

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value
