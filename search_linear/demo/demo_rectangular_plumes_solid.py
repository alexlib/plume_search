from __future__ import division, print_function
import matplotlib.pyplot as plt

import search_agent
import hit_probability_functions
import simulation
from config.demo_rectangular_plumes_solid import *


def main():
    hit_probability_function = hit_probability_functions.uniform_box_solid
    sim = simulation.Simulation(hit_probability_function, PARAMS,
                                SRC_DENSITY, SEARCH_TIME_MAX, DT,
                                plume_map_resolution=(500, 500))

    agent1 = search_agent.LinearSearcher(theta=THETA, speed=SPEED)
    sim.agent = agent1

    sim.set_src_positions('random')

    _, ax = plt.subplots(1, 1, facecolor='white')
    sim.run(with_plot=True, ax=ax, draw_every=20)

    agent2 = search_agent.LinearSearcher(theta=(THETA + 3 * np.pi/4), speed=SPEED)

    sim.reset()
    sim.agent = agent2
    sim.run(with_plot=True, ax=ax, draw_every=20)
    plt.show(block=True)


if __name__ == '__main__':
    main()