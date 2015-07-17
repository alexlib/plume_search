from __future__ import print_function, division
import unittest
import numpy as np
import matplotlib.pyplot as plt

import search_agent

LEVY_INDEX = .5
DT = .1
PATH_DURATION_MAX = 10
SPEED = 0.5

class LinearSearcherTestCase(unittest.TestCase):
    """
    Test all things related to linear searcher.
    """

    def test_levy_agent(self):

        agent = search_agent.LevySearcher2D(LEVY_INDEX, SPEED, DT, PATH_DURATION_MAX)

        traj = []
        for step in range(int(PATH_DURATION_MAX / DT)):
            traj += [agent.pos.copy()]
            agent.move()

        x, y = np.array(traj).T

        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    unittest.main()