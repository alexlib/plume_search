import numpy as np
#from logprob_odor import advec_diff_mean_hit_rate


def uniform_box_solid(dx, dy, dim_x, dim_y):
    """Return 1 if relative position within box, zero otherwise."""

    within_bdry_x = np.array(np.abs(dx) < dim_x).astype(int)
    within_bdry_y = np.array(np.abs(dy) < dim_y).astype(int)

    return within_bdry_x * within_bdry_y


def uniform_circle_solid(dx, dy, r):
    """Return 1 if relative position within circle, zero otherwise."""

    if np.all(dx**2 + dy**2 <= r**2):
        return 1
    else:
        return 0


def uniform_box_probabilistic(dx, dy, dim_x, dim_y, p):
    """Return p if relative position with box, zero otherwise."""

    within_bdry_x = np.array(np.abs(dx) < dim_x).astype(int)
    within_bdry_y = np.array(np.abs(dy) < dim_y).astype(int)

    return p * (within_bdry_x * within_bdry_y)


def gaussian_concentration(dx, dy, r, d, w, tau):
    """
    Return concentration at a displacement from a "gaussian" plume source.
    Note that particles are assumed to have finite lifetime given by tau.
    :param dx: x-displacement from source
    :param dy: y-displacement from source
    :param r: source emission rate
    :param d: diffusivity
    :param w: windspeed
    :param tau: particle lifetime (s)
    :return:
    """
    if isinstance(dx, (int, long, float)):
        if dx == 0 and dy == 0:
            c = np.inf
        elif dx <= 0 or dx > w * tau:
            c = 0.
        else:
            norm_factor = r / (2 * np.sqrt(np.pi * d * dx))
            exp_factor = np.exp(-w * dy**2 / (4 * d * dx))
            c = norm_factor * exp_factor
    else:
        norm_factor = r / (2 * np.sqrt(np.pi * d * dx))
        exp_factor = np.exp(-w * dy**2 / (4 * d * dx))
        c = norm_factor * exp_factor
        c[dx <= 0] = 0.
        c[(dx == 0) * (dy == 0)] = np.inf

    return c


def gaussian_solid(dx, dy, r, d, w, th):
    """
    Return 1 if concentration is greater than threshold.
    Plume assumes wind blowing from -x to +x
    :param dx: x-displacement from source
    :param dy: y-displacement from source
    :param d: diffusivity
    :param w: windspeed
    :param r: source emission rate
    :param th: threshold
    :return: 1 if concentration at dx, dy > threshold, 0 otherwise (nan is returned if dx is negative [uw of source])
    """

    c = np.array(gaussian_concentration(dx, dy, r, d, w))

    return (c > th).astype(float)


def gaussian_probabilistic(dx, dy, dt, r, d, w):
    """
    Return 1 if concentration is greater than threshold.
    Plume assumes wind blowing from -x to +x
    :param dx: x-displacement from source
    :param dy: y-displacement from source
    :param d: diffusivity
    :param w: windspeed
    :param r: source emission rate
    :param th: threshold
    :return: 1 if concentration at dx, dy > threshold, 0 otherwise (nan is returned if dx is negative [uw of source])
    """

    c = gaussian_concentration(dx, dy, r, d, w)

    return 1 - np.exp(-c*dt)