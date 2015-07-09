from __future__ import print_function, division
import unittest
import numpy as np

import search_agent


class LinearSearcherTestCase(unittest.TestCase):
    """
    Test all things related to linear searcher.
    """

    def test_agent_starts_at_0_0(self):

        for _ in range(10):
            theta = np.random.uniform(-np.pi, np.pi)
            speed = np.random.uniform(0, 10)
            agent = search_agent.LinearSearcher(theta=theta, speed=speed)
            self.assertEqual(tuple(agent.pos), (0., 0.))

    def test_moves_in_correct_direction(self):

        for _ in range(10):
            theta = np.random.uniform(-np.pi, np.pi)
            speed = np.random.uniform(0, 10)
            agent = search_agent.LinearSearcher(theta=theta, speed=speed)

            for dt in np.arange(.1, 1.1, .1):
                dr_correct = dt * speed * np.array([np.cos(theta), np.sin(theta)])
                pos_next_correct = agent.pos + dr_correct
                agent.move(dt)

                self.assertAlmostEqual(agent.pos[0], pos_next_correct[0], places=7)
                self.assertAlmostEqual(agent.pos[1], pos_next_correct[1], places=7)


if __name__ == '__main__':
    unittest.main()