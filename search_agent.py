from __future__ import print_function, division
import numpy as np


class LinearSearcher(object):
    """
    Search agent that moves only in a straight line.
    """

    def __init__(self, theta, speed):
        self.theta = theta
        self.speed = speed
        self.vx = self.speed * np.cos(self.theta)
        self.vy = self.speed * np.sin(self.theta)

        self.pos = None
        self.reset()

    def move(self, dt):
        self.pos += np.array([self.vx * dt, self.vy * dt])

    def reset(self):
        self.pos = np.array([0., 0])

    @staticmethod
    def detect_odor(hit_prob):
        return np.random.rand() < hit_prob