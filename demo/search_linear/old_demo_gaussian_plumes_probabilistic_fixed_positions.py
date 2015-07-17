from __future__ import division, print_function

import matplotlib.pyplot as plt

import search_agent
import hit_probability_functions
import simulation_old
from config.demo_gaussian_plumes_probabilistic_fixed_positions import *


PARAMS = {'dt': DT,
          'r': .02,
          'd': 0.02,
          'w': 0.5}

def main():
    hit_probability_function = hit_probability_functions.gaussian_probabilistic
    sim = simulation_old.Simulation(hit_probability_function, PARAMS,
                                    SRC_DENSITY, SEARCH_TIME_MAX, DT,
                                    plume_map_resolution=(500, 500))

    agent1 = search_agent.LinearSearcher(theta=THETAS[1], speed=SPEED)
    sim.agent = agent1

    sim.set_src_positions(SRC_POSITIONS)

    _, ax = plt.subplots(1, 1, facecolor='white')
    sim.run(with_plot=True, ax=ax, draw_every=20)

    for theta in THETAS[2:3]:
        print('\n\nbreak\n\n')
        agent2 = search_agent.LinearSearcher(theta=theta, speed=SPEED)

        sim.reset()
        sim.agent = agent2
        sim.run(with_plot=True, ax=ax, draw_every=20)

    plt.show(block=True)

if __name__ == '__main__':
    main()