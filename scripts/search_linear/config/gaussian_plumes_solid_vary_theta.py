from __future__ import division
import numpy as np

SRC_DENSITY = 0.005  # per m^2
SEARCH_TIME_MAX = 20  # s
SPEED = 0.5  # m/s
DT = .1
AGENT_SEARCH_RADIUS = SEARCH_TIME_MAX * DT
SRC_POSITIONS = 'random'

# PLUME STRUCTURE PARAMETERS
PARAMS_PLUME_STRUCTURE = {'r': 0.02,
                          'd': 0.02,
                          'w': 0.5,
                          'threshold': .01,
                          'tau': 24,
                          'q': .0001}

# LOOPING PARAMETERS
N_ENVIRONMENTS = 300
THETAS = np.linspace(-np.pi, np.pi, 33)