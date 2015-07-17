from __future__ import print_function, division
import unittest
import numpy as np

import plume_structures


class GaussianPlumeModelsTestCase(unittest.TestCase):

    def setUp(self):
        print('In method "{}"...'.format(self._testMethodName))

        self.params = {'r': .005,
                       'd': .02,
                       'w': 0.5,
                       'tau': 12,
                       'q': .0001}
        self.plume_structure = plume_structures.Gaussian2D(**self.params)

    def test_gaussian_concentration_smaller_downwind_of_source(self):
        c_uw = self.plume_structure.conc(.1, 0)
        c_dw = self.plume_structure.conc(.2, 0)
        self.assertLess(c_dw, c_uw)
        self.assertGreater(c_dw, 0)
        self.assertGreater(c_uw, 0)

    def test_gaussian_concentration_is_inf_at_source(self):
        c = self.plume_structure.conc(0, 0)
        self.assertTrue(np.isinf(c))

    def test_gaussian_concentration_is_zero_outside_of_boundaries(self):
        c_uw = self.plume_structure.conc(-.1, 0)
        dx_dw = self.plume_structure.bdry[1] + .1
        dy_cw = self.plume_structure.bdry[2] + .1
        c_dw = self.plume_structure.conc(dx_dw, 0)
        c_cw = self.plume_structure.conc(.5, dy_cw)

        for c in c_uw, c_dw, c_cw:
            self.assertEqual(c, 0)

    def test_arrays_are_interpreted_correctly(self):
        dx = np.array([-.2, -.1, 0., 0, .1, .2])
        dy = np.array([.1, .1, 0, .1, .3, -.9])

        c = self.plume_structure.conc(dx, dy)

        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 0)
        self.assertTrue(np.isinf(c[2]))
        self.assertEqual(c[3], 0)

        for idx in range(4, 6):
            self.assertGreater(c[idx], 0)


if __name__ == '__main__':
    unittest.main()