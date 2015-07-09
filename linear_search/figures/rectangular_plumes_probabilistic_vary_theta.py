from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
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


plume_found_prob = plume_found.sum(0) / N_ENVIRONMENTS
search_times_mean = np.nanmean(search_times, axis=0)
search_times_std = np.nanstd(search_times, axis=0)

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(THETAS * 180 / np.pi, plume_found_prob, lw=2)
axs[1].errorbar(THETAS * 180 / np.pi, search_times_mean, yerr=search_times_std, lw=2)

axs[0].set_xlim(-180, 180)
axs[1].set_xlabel('theta')
axs[0].set_ylabel('P(found plume)')
axs[1].set_ylabel('T(found plume)')

plt.show(block=True)