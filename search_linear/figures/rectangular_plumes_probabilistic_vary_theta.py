from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from math_tools import stats
import search_agent
import simulation

from config.rectangular_plumes_probabilistic_vary_theta import *

plume_found = np.zeros((N_ENVIRONMENTS, len(THETAS)))
search_times = np.nan * np.ones((N_ENVIRONMENTS, len(THETAS)), dtype=float)

for e_ctr in range(N_ENVIRONMENTS):
    print(e_ctr)
    sim = simulation.Simulation(HIT_PROBABILITY_FUNCTION, PARAMS,
                                SRC_DENSITY, SEARCH_TIME_MAX, DT)

    for th_ctr, theta in enumerate(THETAS):
        agent = search_agent.LinearSearcher(theta=theta, speed=SPEED)

        sim.reset()
        sim.agent = agent

        if th_ctr == 0:
            sim.set_src_positions('random')

        sim.run()
        plume_found[e_ctr, th_ctr] = int(sim.plume_found)
        if sim.plume_found:
            search_times[e_ctr, th_ctr] = sim.search_time


plume_found_n = plume_found.sum(0)
plume_found_prob = plume_found_n / N_ENVIRONMENTS
search_times_mean = np.nanmean(search_times, axis=0)
search_times_std = np.nanstd(search_times, axis=0)

fig, ax = plt.subplots(1, 1, sharex=True)
lbs, ubs = np.transpose([stats.binomial_confidence_conjugate_prior(n, N_ENVIRONMENTS) for n in plume_found_n])
err_lower = plume_found_prob - lbs
err_upper = ubs - plume_found_prob
ax.errorbar(THETAS * 180 / np.pi, plume_found_prob, yerr=[err_lower, err_upper], lw=2)

ax.set_xlim(-180, 180)
ax.set_xticks(np.linspace(-180, 180, 9))
ax.set_ylim(0, 1)

ax.set_xlabel('heading (degrees')
ax.set_ylabel('P(found plume)')

ax.set_title('P_hit = {}'.format(PARAMS['p']))

plt.show(block=True)