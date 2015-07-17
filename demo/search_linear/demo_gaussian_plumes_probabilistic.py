from __future__ import division, print_function

import matplotlib.pyplot as plt

import environment
import search_agent
import plume_structures
import simulation

from config.demo_gaussian_plumes_probabilistic import *


def main():
    plume_structure = plume_structures.Gaussian2D(**PARAMS)

    agent_search_radius = SPEED * SEARCH_TIME_MAX

    env = environment.Environment2d(plume_structure, SRC_DENSITY, agent_search_radius)

    _, ax = plt.subplots(1, 1, facecolor='white')

    draw_background = True
    for theta in THETAS:
        agent = search_agent.LinearSearcher(theta=theta, speed=SPEED)
        trial = simulation.Trial2d(env, agent, SEARCH_TIME_MAX, DT)
        trial.run(with_plot=True, ax=ax, draw_every=20, draw_background=draw_background)
        draw_background = False

    plt.show(block=True)


if __name__ == '__main__':
    main()