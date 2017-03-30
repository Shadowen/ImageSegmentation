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
