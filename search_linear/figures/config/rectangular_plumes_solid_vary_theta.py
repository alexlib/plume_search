from __future__ import division
import numpy as np
import hit_probability_functions

SRC_DENSITY = 0.02  # per m^2
SEARCH_TIME_MAX = 20  # s
SPEED = 0.5  # m/s
THETA = np.pi / 4
DT = .1

# HIT PROBABILITY
HIT_PROBABILITY_FUNCTION = hit_probability_functions.uniform_box_solid
PARAMS = {'dim_x': 3,
          'dim_y': .1}

# LOOPING PARAMETERS
N_ENVIRONMENTS = 100
THETAS = np.linspace(-np.pi, np.pi, 32, endpoint=False)