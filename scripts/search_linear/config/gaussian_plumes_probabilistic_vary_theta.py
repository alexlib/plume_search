from __future__ import division
import numpy as np
import hit_probability_functions

SRC_DENSITY = 0.01  # per m^2
SEARCH_TIME_MAX = 80  # s
SPEED = 0.5  # m/s
DT = .1

# HIT PROBABILITY
HIT_PROBABILITY_FUNCTION = hit_probability_functions.gaussian_probabilistic
PARAMS = {'dt': DT,
          'r': .02,
          'd': 0.02,
          'w': 0.5}

# LOOPING PARAMETERS
N_ENVIRONMENTS = 100
THETAS = np.linspace(-np.pi, np.pi, 33)