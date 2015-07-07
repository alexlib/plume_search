import numpy as np

# VARIABLE PARAMETERS
THETAS = np.linspace(-np.pi, np.pi, 64, endpoint=False)
DENSITIES = np.array([.1, .5, 1, 2, 5])
V_WINDS = np.array([0, 0.5])  # m/s

# FIXED PARAMETERS
# insect parameters
V_INSECT = 0.2  # m/s

# plume parameters
A = 0.002  # m
R = 100  # conc m^2/s
D = 0.1  # m^2/s
TAU = 10000  # seconds

# simulation parameters
N_TRIALS_PER_THETA = 100
DT = 0.1 # s
T_MAX = 60  # s

# calculated parameters
N_STEPS_MAX = int(round(T_MAX / DT))
N_THETAS = len(THETAS)