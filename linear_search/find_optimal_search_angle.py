"""
Let an agent fly in a straight line through a large environment containing multiple odor plumes. Calculate the
distribution of times till first odor hit detection for a given search angle.
"""
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import logprob_odor

from config.find_optimal_search_angle import *

# PLOTTING PARAMETERS
N_ROWS = len(V_WINDS)
N_COLS = len(DENSITIES)
FIG_SIZE = (12, 6)
FACE_COLOR = 'white'
LW = 2


class SearchAgent(object):
    """
    Search agent capable of moving in a specified way.
    """

    def __init__(self, v, theta, dt):
        self.v = v
        self.theta = theta
        self.dt = dt

        dx = self.v * self.dt * np.cos(self.theta)
        dy = self.v * self.dt * np.sin(self.theta)
        self.dr = np.array([dx, dy])

        self.pos = np.array([0, 0.])

    def move(self):
        self.pos += self.dr

    def detect_odor(self, mean_hit_rate):
        mean_hit_num = self.dt * mean_hit_rate
        return np.random.poisson(mean_hit_num) > 0


def calc_mean_hit_rate(pos, src_poss, wind):
    """
    Calculate the total mean hit rate at position 'pos' given sources in positions src_poss.
    :param pos: position
    :param src_poss: list of source positions
    :param wind: wind speed (m/s)
    :return: mean hit rate
    """

    mean_hit_rate = 0
    for src_pos in src_poss:
        # calculate mean hit rate from this source
        dx = pos[0] - src_pos[0]
        dy = pos[1] - src_pos[1]
        dz = 0

        r = logprob_odor.advec_diff_mean_hit_rate(dx, dy, dz, w=wind, r=R, d=D, a=A, tau=TAU, dim=2)
        mean_hit_rate += r

    return mean_hit_rate


def main():
    x_max_insect = N_STEPS_MAX * V_INSECT
    x_min_insect = -x_max_insect
    y_max_insect = N_STEPS_MAX * V_INSECT
    y_min_insect = -y_max_insect

    n_steps_distr = np.nan * np.ones((N_ROWS, N_COLS, N_TRIALS_PER_THETA, N_THETAS), dtype=float)

    for v_wind_ctr, v_wind in enumerate(V_WINDS):
        # TODO determine envelope of plume
        x_max_src = x_max_insect + x_src_size_up
        x_min_src = x_min_insect - x_src_size_down
        y_max_src = y_max_insect + y_src_size
        y_min_src = y_min_insect - y_src_size

        src_distr_area = (y_max_src - y_min_src) * (x_max_src - x_min_src)
        for d_ctr, density in enumerate(DENSITIES):

            n_srcs = np.random.poisson(src_distr_area * density)
            src_poss = np.random.uniform([x_min_src, x_max_src],
                                         [y_min_src, y_max_src],
                                         (n_srcs, 2))

            for th_ctr, theta in enumerate(THETAS):
                for tr_ctr in range(N_TRIALS_PER_THETA):
                    agent = SearchAgent(v=V_INSECT, theta=theta, dt=DT)

                    for step in xrange(N_STEPS_MAX):
                        mean_hit_rate = calc_mean_hit_rate(agent.pos, src_poss)
                        if agent.detect_odor(mean_hit_rate):
                            n_steps = step
                            break
                        else:
                            agent.move()
                    else:
                        n_steps = -1

                    n_steps_distr[v_wind_ctr, d_ctr, tr_ctr, th_ctr] = n_steps

    # plot source finding probability
    fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=FIG_SIZE, facecolor=FACE_COLOR,
                            subplot_kw={'polar': True}, tight_layout=True)

    for v_wind_ctr, v_wind in enumerate(V_WINDS):
        for d_ctr, density in enumerate(DENSITIES):
            ax = axs[v_wind_ctr, d_ctr]
            t = THETAS
            r = np.sum(~np.isnan(n_steps_distr[v_wind_ctr, d_ctr]), axis=0) / N_TRIALS_PER_THETA

            ax.plot(t, r, lw=LW)

    # plot search time distributions
    fig, axs = plt.subplots(N_ROWS, N_COLS, figsize=FIG_SIZE, facecolor=FACE_COLOR,
                            subplot_kw={'polar': True}, tight_layout=True)

    for v_wind_ctr, v_wind in enumerate(V_WINDS):
        for d_ctr, density in enumerate(DENSITIES):
            ax = axs[v_wind_ctr, d_ctr]
            t = THETAS
            r = np.nanmean(n_steps_distr[v_wind_ctr, d_ctr], axis=0)
            yerr = np.nanstd(n_steps_distr[v_wind_ctr, d_ctr], axis=0)

            ax.errorbar(t, r, lw=LW, yerr=yerr)

    plt.show()


if __name__ == '__main__':
    main()