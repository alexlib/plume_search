from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt


class Environment2d(object):
    """
    Two-dimensional plume-containing environment.
    """

    def __init__(self, plume_structure, src_density, agent_search_radius, src_positions='random'):

        self.plume_structure = plume_structure
        self.src_density = src_density
        self.agent_search_radius = agent_search_radius

        # calculate relevant area to distribute plumes within
        self.bdry = [-agent_search_radius - plume_structure.bdry[1],
                     agent_search_radius + plume_structure.bdry[0],
                     -agent_search_radius - plume_structure.bdry[3],
                     agent_search_radius + plume_structure.bdry[2]]
        self.area = (self.bdry[1] - self.bdry[0]) * (self.bdry[3] - self.bdry[2])

        self.src_positions = None
        self.set_src_positions(src_positions)

    def set_src_positions(self, src_positions):
        """
        Set positions of all sources.
        :param src_positions: 'random' or N x 2 array for N sources
        """
        if src_positions == 'random':
            n_srcs = np.random.poisson(self.area * self.src_density)
            self.src_positions = np.random.uniform([self.bdry[0], self.bdry[2]],
                                                   [self.bdry[1], self.bdry[3]],
                                                   size=(n_srcs, 2))
        else:
            if not isinstance(src_positions, np.ndarray):
                raise TypeError('"src_positions" must be an N x 2 numpy array!')
            elif src_positions.ndim != 2:
                raise TypeError('"src_positions" must be an N x 2 numpy array!')

            self.src_positions = src_positions

    def miss_probability(self, x, y, dt):

        miss_probability = np.ones(np.array(x).shape, dtype=float)

        for src_x, src_y in self.src_positions:
            miss_probability *= self.plume_structure.miss_probability(x - src_x, y - src_y, dt)

        return miss_probability

    def hit_probability(self, x, y, dt):

        return 1 - self.miss_probability(x, y, dt)

    def sample(self, x, y, dt):

        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            raise TypeError('"x" and "y" cannot be arrays!')

        print(self.hit_probability(x, y, dt))
        return int(np.random.rand() < self.hit_probability(x, y, dt))

    def heatmap(self, resolution=(500, 500)):
        """
        Compute the 2D heatmap of the environment.
        :param resolution:
        :return: heatmap, extent

        Note: this returns a matrix whose rows correspond to x and whose columns correspond to y. To properly plot
        the heatmap, be sure to use .matshow(heatmap.T, origin='lower', extent=extent)
        """

        x = np.linspace(self.bdry[0], self.bdry[1], num=resolution[0])
        y = np.linspace(self.bdry[2], self.bdry[3], num=resolution[1])
        x_m, y_m = np.meshgrid(x, y, indexing='ij')

        hit_probability = 1 - self.miss_probability(x_m, y_m, dt=1)

        extent = [self.bdry[0] - .5*(x[1] - x[0]),
                  self.bdry[1] + .5*(x[1] - x[0]),
                  self.bdry[2] - .5*(y[1] - y[0]),
                  self.bdry[3] + .5*(y[1] - y[0])]

        return hit_probability, extent