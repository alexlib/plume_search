from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

import search_agent
import hit_probability_functions
import simulation
from config.demo_rectangular_plumes_solid import *


def main():
    agent = search_agent.LinearSearcher(theta=np.pi/2, speed=SPEED)
    hit_probability_function = hit_probability_functions.uniform_box
    sim = simulation.Simulation(agent, hit_probability_function, PARAMS,
                                SRC_DENSITY, SEARCH_TIME_MAX, DT)

    sim.set_src_positions('random')

    _, ax = plt.subplots(1, 1)
    sim.run_with_plot(ax, draw_every=20)

    plt.show(block=True)


if __name__ == '__main__':
    main()