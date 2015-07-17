from __future__ import print_function, division
import matplotlib.pyplot as plt
from math_tools import stats

import search_agent
import simulation
import plume_structures
import environments

from config.gaussian_plumes_probabilistic_vary_theta import *


plume_structure = plume_structures.Gaussian2D(**PARAMS_PLUME_STRUCTURE)
agents = [search_agent.LinearSearcher(theta=theta, speed=SPEED) for theta in THETAS]
envs = []

sim = simulation.Simulation(plume_structure, agents=agents, n_environments=N_ENVIRONMENTS)

plume_detected = np.zeros((N_ENVIRONMENTS, len(THETAS)))
search_times = np.nan * np.ones((N_ENVIRONMENTS, len(THETAS)), dtype=float)

for e_ctr in range(N_ENVIRONMENTS):
    print(e_ctr)

    # make new environment
    env = environments.Environment2d(plume_structure, SRC_DENSITY, AGENT_SEARCH_RADIUS, SRC_POSITIONS)
    envs += [env]
    for a_ctr, agent in enumerate(agents):
        # set agent's starting position back to zero
        agent.reset()

        trial = simulation.Trial2d(env, agent, SEARCH_TIME_MAX, DT)
        trial.run()

        if trial.plume_detected:
            plume_detected[e_ctr, a_ctr] = 1


plume_detected_n = plume_detected.sum(0)
plume_detected_prob = plume_detected.sum(0) / N_ENVIRONMENTS

fig, ax = plt.subplots(1, 1, facecolor='white')
lbs, ubs = np.transpose([stats.binomial_confidence_conjugate_prior(n, N_ENVIRONMENTS) for n in plume_detected_n])
err_lower = plume_detected_prob - lbs
err_upper = ubs - plume_detected_prob
ax.errorbar(THETAS * 180 / np.pi, plume_detected_prob, yerr=[err_lower, err_upper], lw=2)

ax.set_xlim(-180, 180)
ax.set_xticks(np.linspace(-180, 180, 9))
ax.set_ylim(0, 1)

ax.set_xlabel('heading (degrees)')
ax.set_ylabel('P(plume detected)')

ax.set_title('probabilistic')

plt.show(block=True)