from __future__ import division, print_function
import unittest
import numpy as np

import environments
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
        self.src_density = 0.1
        self.agent_search_radius = 10
        self.env = environments.Environment2d(self.plume_structure,
                                             self.src_density,
                                             self.agent_search_radius,
                                             src_positions='random')
        src_positions_single = np.array([[-1., 1]])
        self.env_single_src = environments.Environment2d(self.plume_structure,
                                                        self.src_density,
                                                        self.agent_search_radius,
                                                        src_positions=src_positions_single)

    def test_miss_probability_is_between_0_and_1_at_grid_of_positions(self):
        x = np.linspace(self.env.bdry[0], self.env.bdry[-1], 20)
        y = np.linspace(self.env.bdry[2], self.env.bdry[3], 20)
        x_m, y_m = np.meshgrid(x, y, indexing='ij')

        for xx, yy in zip(x_m.flatten(), y_m.flatten()):
            miss_prob = self.env.miss_probability(xx, yy, dt=.1)
            self.assertGreaterEqual(miss_prob, 0)
            self.assertLessEqual(miss_prob, 1)

    def test_hit_probability_increases_with_dt(self):

        hit_prob_dt_small = self.env_single_src.hit_probability(0, 0, dt=.1)
        hit_prob_dt_large = self.env_single_src.hit_probability(0, 0, dt=.3)

        self.assertGreater(hit_prob_dt_large, hit_prob_dt_small)

    def test_hit_probability_is_zero_uw_of_src(self):

        hit_prob_uw = self.env_single_src.hit_probability(-1.1, 0, dt=.1)
        self.assertEqual(hit_prob_uw, 0)


if __name__ == '__main__':
    unittest.main()