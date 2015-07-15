from __future__ import division
import numpy as np

SRC_DENSITY = 0.03  # per m^2
SEARCH_TIME_MAX = 20  # s
SPEED = 0.5  # m/s
THETAS = np.linspace(0, 2*np.pi, 8, endpoint=False)
DT = .1

# HIT PROBABILITY PARAMS
PARAMS = {'dt': DT,
          'r': .02,
          'd': 0.02,
          'w': 0.5}