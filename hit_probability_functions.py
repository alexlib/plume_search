import numpy as np
from logprob_odor import advec_diff_mean_hit_rate


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


def gaussian_concentration(dx, dy, r, d, w):
    """
    Return concentration at a displacement from a "gaussian" plume source.
    :param dx: x-displacement from source
    :param dy: y-displacement from source
    :param d: diffusivity
    :param w: windspeed
    :param r: source emission rate
    :return:
    """
    norm_factor = r / (2 * np.sqrt(np.pi * d * dx / w))
    exp_factor = np.exp(-w * dy**2 / (4 * d * dx))

    c = np.array(norm_factor * exp_factor)

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

    c = gaussian_concentration(dx, dy, r, d, w)

    if c.shape:
        # if c is a non-zero sized array
        c[np.isnan(c)] = 0
    else:
        if np.isnan(c):
            c = np.array(0)

    return (c > th).astype(int)


def gaussian_probabilistic(dx, dy, r, d, w):
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

    if c.shape:
        # if c is a non-zero sized array
        c[np.isnan(c)] = 0
    else:
        if np.isnan(c):
            c = np.array(0)

    return 1 - np.exp(-c)