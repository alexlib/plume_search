from __future__ import division
import numpy as np

SRC_DENSITY = 0.04  # per m^2
SEARCH_TIME_MAX = 3  # s
SPEED = 0.5  # m/s
THETAS = np.linspace(0, 2*np.pi, 8, endpoint=False)
DT = .1

# HIT PROBABILITY PARAMS
PARAMS = {'r': 0.02,
          'd': 0.02,
          'w': 0.5,
          'tau': 24,
          'q': .0001}

SRC_POSITIONS = np.array([[-6., 0],
                          [-6, 6],
                          [-6, -6]])