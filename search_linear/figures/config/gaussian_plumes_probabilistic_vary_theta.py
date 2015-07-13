from __future__ import division
import numpy as np
import hit_probability_functions

SRC_DENSITY = 0.02  # per m^2
SEARCH_TIME_MAX = 20  # s
SPEED = 0.5  # m/s
THETA = np.pi / 4
DT = .1

# HIT PROBABILITY
HIT_PROBABILITY_FUNCTION = hit_probability_functions.gaussian_probabilistic
PARAMS = {'r': .005,
          'd': 0.02,
          'w': 0.5}

# LOOPING PARAMETERS
N_ENVIRONMENTS = 1000
THETAS = np.linspace(-np.pi, np.pi, 32, endpoint=False)