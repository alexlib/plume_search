from __future__ import print_function, division
import unittest
import numpy as np

import hit_probability_functions


class HitRateFunctionsTestCase(unittest.TestCase):

    def test_hit_rate_function_uniform_box_gives_correct_output(self):

        func = hit_probability_functions.uniform_box
        dims = [1, 2]

        self.assertEqual(func(0, 1, *dims), 1)
        self.assertEqual(func(1, 3, *dims), 0)

    def test_hit_rate_function_uniform_circle_gives_correct_output(self):

        func = hit_probability_functions.uniform_circle

        self.assertEqual(func(0, 1, 2), 1)
        self.assertEqual(func(5, 3, 2), 0)


if __name__ == '__main__':
    unittest.main()