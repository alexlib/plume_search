from __future__ import print_function, division
import numpy as np


class Searcher(object):

    def reset(self):
        self.pos = np.array([0., 0])

    @staticmethod
    def detect_odor(hit_prob):
        return np.random.rand() < hit_prob


class LinearSearcher(Searcher):
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


class RandomSearcher(Searcher):
    """
    Search agent that moves in steps of random direction.
    """

    def __init__(self, speed):
        self.speed = speed

        self.pos = None
        self.reset()

    def move(self, dt):
        theta = np.random.uniform(-np.pi, np.pi)
        dx = self.speed * dt * np.cos(theta)
        dy = self.speed * dt * np.sin(theta)

        self.pos += np.array([dx, dy])


class LevySearcher2D(Searcher):
    """
    Search agent that moves along levy-flight path.
    """

    def __init__(self, levy_index, speed, dt, path_duration_max):
        self.levy_index = levy_index
        self.speed = speed
        self.dt = dt
        self.path_duration_max = path_duration_max
        self.step_size = self.dt * self.speed
        self.path_duration_max_int = round(path_duration_max / dt)

        self.pos = None
        self.sample_next_path = True
        self.theta = None
        self.steps_till_next_sample = -1
        self.reset()

    def reset(self):
        self.pos = np.array([0., 0])
        self.sample_next_path = True
        self.theta = None

    def move(self, dt):
        if not self.sample_next_path:
            # take another step along the path
            dr = self.step_size * np.array([np.cos(self.theta), np.sin(self.theta)])
            self.pos += dr
        else:
            # sample direction uniformly
            self.theta = np.random.uniform(0, 2*np.pi)
            # sample path length from power law distribution
            path_lengths = np.arange(1, self.path_duration_max_int + 1, dtype=float)
            prob = path_lengths ** -self.levy_index
            prob /= prob.sum()

            self.steps_till_next_sample = np.random.choice(path_lengths, p=prob)
            print('{} steps till next sample'.format(self.steps_till_next_sample))
            self.sample_next_path = False

        self.steps_till_next_sample -= 1
        if self.steps_till_next_sample == 0:
            self.sample_next_path = True