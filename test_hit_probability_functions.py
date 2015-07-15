from __future__ import print_function, division
import unittest
import numpy as np

import hit_probability_functions


class BasicHitRateFunctionsTestCase(unittest.TestCase):

    def test_uniform_box_gives_correct_output(self):

        func = hit_probability_functions.uniform_box_solid
        dims = [1, 2]

        self.assertEqual(func(0, 1, *dims), 1)
        self.assertEqual(func(1, 3, *dims), 0)

    def test_uniform_circle_gives_correct_output(self):

        func = hit_probability_functions.uniform_circle_solid

        self.assertEqual(func(0, 1, 2), 1)
        self.assertEqual(func(5, 3, 2), 0)


class GaussianPlumeModelsTestCase(unittest.TestCase):

    def setUp(self):
        self.params = {'r': .005,
                       'd': .02,
                       'w': 0.5}
        print('In method "{}"'.format(self._testMethodName))

    def test_gaussian_concentration_smaller_downwind_of_source(self):
        c_uw = hit_probability_functions.gaussian_concentration(.1, 0, **self.params)
        c_dw = hit_probability_functions.gaussian_concentration(.2, 0, **self.params)
        self.assertLess(c_dw, c_uw)
        self.assertGreater(c_dw, 0)
        self.assertGreater(c_uw, 0)

    def test_gaussian_concentration_is_inf_at_source(self):
        c = hit_probability_functions.gaussian_concentration(0., 0, **self.params)
        self.assertTrue(np.isinf(c))

    def test_gaussian_concentration_is_zero_uw_of_source(self):
        c_uw = hit_probability_functions.gaussian_concentration(-.1, 0, **self.params)
        self.assertEqual(c_uw, 0)

    def test_arrays_are_interpreted_correctly(self):
        dx = np.array([-.2, -.1, 0., 0, .1, .2])
        dy = np.array([.1, .1, 0, .1, .3, -.9])

        c = hit_probability_functions.gaussian_concentration(dx, dy, **self.params)

        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 0)
        self.assertTrue(np.isinf(c[2]))
        self.assertEqual(c[3], 0)

        for idx in range(4, 6):
            self.assertGreater(c[idx], 0)

    def test_hit_probability_is_one_at_src(self):
        p = hit_probability_functions.gaussian_probabilistic(0, 0, **self.params)
        self.assertEqual(p, 1)

    def test_hit_probability_is_zero_uw_and_cw_from_src_and_positive_dw_of_src(self):
        p_uw = hit_probability_functions.gaussian_probabilistic(-.1, 0, **self.params)
        p_cw1 = hit_probability_functions.gaussian_probabilistic(0, .1, **self.params)
        p_cw2 = hit_probability_functions.gaussian_probabilistic(0, -.1, **self.params)
        p_dw = hit_probability_functions.gaussian_probabilistic(.1, 0., **self.params)

        self.assertEqual(p_uw, 0)
        self.assertEqual(p_cw1, 0)
        self.assertEqual(p_cw2, 0)
        self.assertGreater(p_dw, 0)

    def hit_probability_is_larger_closer_to_src_if_dw_of_src(self):
        p_uw = hit_probability_functions.gaussian_probabilistic(.1, 0, **self.params)
        p_dw = hit_probability_functions.gaussian_probabilistic(.2, 0, **self.params)

        self.assertGreater(p_uw, p_dw)


if __name__ == '__main__':
    unittest.main()