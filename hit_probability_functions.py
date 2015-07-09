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