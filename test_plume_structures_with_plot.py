from __future__ import print_function, division
import unittest
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.ion()

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


    def test_heatmap_generated_correctly(self):
        heatmap, extent = self.plume_structure.heatmap(resolution=(500, 500))
        plt.matshow(heatmap.T, origin='lower', extent=extent, cmap=cm.hot)
        plt.draw()
        x = raw_input('Was this plot generated correctly [y/n]?')
        self.assertEqual(x[0].lower(), 'y')


if __name__ == '__main__':
    unittest.main()