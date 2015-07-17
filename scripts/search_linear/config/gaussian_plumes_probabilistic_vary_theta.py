from __future__ import division
import numpy as np
import hit_probability_functions

SRC_DENSITY = 0.04  # per m^2
SEARCH_TIME_MAX = 20  # s
SPEED = 0.5  # m/s
DT = .1
AGENT_SEARCH_RADIUS = SEARCH_TIME_MAX * DT
SRC_POSITIONS = 'random'

# PLUME STRUCTURE PARAMETERS
PARAMS_PLUME_STRUCTURE = {'r': .02,
                          'd': 0.02,
                          'w': 0.5,
                          'tau': 24,
                          'q': .0001}

# LOOPING PARAMETERS
N_ENVIRONMENTS = 100
THETAS = np.linspace(-np.pi, np.pi, 33)