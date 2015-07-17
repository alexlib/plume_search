from __future__ import division, print_function
import unittest
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.ion()

import environment
import plume_structures


class Environment2dGaussianPlumeStructureTestCase(unittest.TestCase):

    def setUp(self):
        print('In method "{}"...'.format(self._testMethodName))

        self.params = {'r': .1,
                       'd': .02,
                       'w': 0.5,
                       'tau': 12,
                       'q': .0001}
        self.plume_structure = plume_structures.Gaussian2D(**self.params)
        self.src_density = 0.01
        self.agent_search_radius = 10
        self.env_random = environment.Environment2d(self.plume_structure, self.src_density, self.agent_search_radius)
        src_positions_fixed = np.array([[-3.1, 0], [5.1, 3.2], [-8.9, -2.7]])
        self.env_fixed = environment.Environment2d(self.plume_structure,
                                                   self.src_density,
                                                   self.agent_search_radius,
                                                   src_positions_fixed)
        src_positions_single = np.array([[5.1, 3.2]])
        self.env_single_src = environment.Environment2d(self.plume_structure,
                                                        self.src_density,
                                                        self.agent_search_radius,
                                                        src_positions_single)

    def test_env_single_src_plot_looks_okay(self):

        heatmap, extent = self.env_single_src.heatmap(resolution=(500, 500))

        plt.matshow(heatmap.T, origin='lower', extent=extent, cmap=cm.hot)
        plt.draw()
        print('This plot should have a single source located at (5.1, 3.2).')
        x = raw_input('Does this plot look correct [y/n]?')
        self.assertEqual(x[0].lower(), 'y')

    def test_env_fixed_plot_looks_okay(self):

        heatmap, extent = self.env_fixed.heatmap(resolution=(500, 500))

        plt.matshow(heatmap.T, origin='lower', extent=extent, cmap=cm.hot)
        plt.draw()
        print('This plot should have three sources located at (-3.1, 0), (5.1, 3.2), and (-8.9, -2.7).')
        x = raw_input('Does this plot look correct [y/n]?')
        self.assertEqual(x[0].lower(), 'y')

    def test_env_random_plot_looks_okay(self):

        heatmap, extent = self.env_random.heatmap(resolution=(500, 500))

        plt.matshow(heatmap.T, origin='lower', extent=extent, cmap=cm.hot)
        plt.draw()
        print('This plot should have several randomly located sources.')
        x = raw_input('Does this plot look correct [y/n]?')
        self.assertEqual(x[0].lower(), 'y')


if __name__ == '__main__':
    unittest.main()