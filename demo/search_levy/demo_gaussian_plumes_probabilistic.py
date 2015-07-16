from __future__ import division, print_function

import matplotlib.pyplot as plt

import search_agent
import hit_probability_functions
import simulation
from demo.search_levy.config.demo_gaussian_plumes_probabilistic import *


def main():
    hit_probability_function = hit_probability_functions.gaussian_probabilistic
    sim = simulation.Simulation(hit_probability_function, PARAMS,
                                SRC_DENSITY, SEARCH_TIME_MAX, DT,
                                plume_map_resolution=(500, 500))

    agent1 = search_agent.LevySearcher2D(levy_index=LEVY_INDEX, speed=SPEED, dt=DT, path_duration_max=PATH_DURATION_MAX)
    sim.agent = agent1

    sim.set_src_positions('random')

    _, ax = plt.subplots(1, 1, facecolor='white')
    sim.run(with_plot=True, ax=ax, draw_every=20)

    plt.show(block=True)

if __name__ == '__main__':
    main()