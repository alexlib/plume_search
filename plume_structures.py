from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


class PlumeStructure(object):

    def heatmap(self, resolution=(500, 500)):
        """
        Compute plume heatmap.
        :param resolution: resolution (pixels x pixels) of heatmap
        :return: heatmap, extent

        Note: this returns a matrix whose rows correspond to x and whose columns correspond to y. To properly plot
        the heatmap, be sure to use .matshow(heatmap.T, origin='lower')
        """

        dx = np.linspace(-self.bdry[0], self.bdry[1], num=resolution[0])
        dy = np.linspace(-self.bdry[2], self.bdry[3], num=resolution[1])
        dx_m, dy_m = np.meshgrid(dx, dy, indexing='ij')

        extent = [-self.bdry[0] - .5*(dx[1] - dx[0]),
                  self.bdry[1] + .5*(dx[1] - dx[0]),
                  -self.bdry[2] + .5*(dy[1] - dy[0]),
                  self.bdry[3] + .5*(dy[1] - dy[0])]

        return self.conc(dx_m, dy_m), extent

    def miss_probability(self, dx, dy, dt):
        """
        Return probability of a miss at a given displacement.
        :param dx: x-displacement from source (positive is downwind of source)
        :param dy: y-displacement from source
        :param dt: time interval over which to integrate concentration
        :return: probability
        """

        return np.exp(-self.conc(dx, dy) * dt)

    def hit_probability(self, dx, dy, dt):
        """
        Return probability of a hit at a given displacement.
        :param dx: x-displacement from source (positive is downwind of source)
        :param dy: y-displacement from source
        :param dt: time interval over which to integrate concentration
        :return: probability
        """

        return 1. - self.miss_probability(dx, dy, dt)


class Gaussian2D(PlumeStructure):
    """
    Gaussian approximation of plume model.
    :param r: source emission rate
    :param d: turbulent diffusivity
    :param w: wind speed (in +x direction)
    :param tau: particle lifetime
    :param q: probability that particle will diffuse past computed boundary
        (this value is used to compute said boundary)
    """

    def __init__(self, r, d, w, tau, q=.0001):
        self.r = r
        self.d = d
        self.w = w
        self.tau = tau
        self.q = q

        # calculate boundary ([x_neg, x_pos, y_neg, y_pos]) (all positive numbers)
        cw_bdry = np.sqrt(-4 * d * tau * np.log(q))
        dw_bdry = w * tau + cw_bdry
        self.bdry = [0, dw_bdry, cw_bdry, cw_bdry]

    def conc(self, dx, dy):
        """
        Calculate concentration a certain displacement from source.
        :param dx: x-displacement from source (positive is downwind of source)
        :param dy: y-displacement from source
        :return:
        """
        if isinstance(dx, (int, long, float)):
            if dx == 0 and dy == 0:
                return np.inf
            elif dx <= 0:
                return 0
            #if dx <= -self.bdry[0] or dx >= self.bdry[1]:
            #    return 0
            #if dy <= -self.bdry[2] or dy >= self.bdry[3]:
            #    return 0

            norm_factor = self.r / (2 * np.sqrt(np.pi * self.d * dx))
            exp_factor = np.exp(-self.w * dy**2 / (4 * self.d * dx))
            return norm_factor * exp_factor
        else:
            norm_factor = self.r / (2 * np.sqrt(np.pi * self.d * dx))
            exp_factor = np.exp(-self.w * dy**2 / (4 * self.d * dx))
            c = norm_factor * exp_factor
            #c[(dx <= -self.bdry[0]) + (dx >= self.bdry[1])] = 0
            #c[(dy <= -self.bdry[2]) + (dy >= self.bdry[3])] = 0
            c[dx <= 0] = 0
            c[(dx == 0)*(dy == 0)] = np.inf
            return c


class GaussianPlume2DSolid(Gaussian2D):
    """
    Solid Gaussian plume (plume detected as soon as agent enters plume envelope).
    :param r: source emission rate
    :param d: turbulent diffusivity
    :param w: wind speed (in +x direction)
    :param tau: particle lifetime
    :param threshold: concentration threshold above which agent detects plume
    :param q: probability that particle will diffuse past computed boundary
        (this value is used to compute said boundary)
    """

    def __init__(self, r, d, w, tau, threshold, q=.0001):
        super(self.__class__, self).__init__(r, d, w, tau, q)
        self.threshold = threshold

    def miss_probability(self, dx, dy):
        """
        Return probability of a miss at a given displacement.
        :param dx: x-displacement from source (positive is downwind of source)
        :param dy: y-displacement from source
        :return: probability
        """

        return np.array(self.c(dx, dy) < self.threshold).astype(float)