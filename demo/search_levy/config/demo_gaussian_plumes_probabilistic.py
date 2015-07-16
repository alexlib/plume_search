from __future__ import division
import numpy as np

SRC_DENSITY = 0.05  # per m^2
SEARCH_TIME_MAX = 20  # s
SPEED = 0.5  # m/s
DT = .1
LEVY_INDEX = 1
PATH_DURATION_MAX = 10

# HIT PROBABILITY PARAMS
PARAMS = {'dt': DT,
          'r': .02,
          'd': 0.02,
          'w': 0.5}