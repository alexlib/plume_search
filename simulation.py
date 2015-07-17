from __future__ import print_function, division
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.ion()


class Trial2d(object):
    """
    Class for running a single agent through an environment.

    :param env: environment instance
    :param agent: agent instance
    :param search_time_max: max amount of time search can go on (s)
    :param dt: timestep (s)
    """

    def __init__(self, env, agent, search_time_max, dt):

        self.env = env
        self.agent = agent
        self.search_time_max = search_time_max
        self.dt = dt

        self.n_steps_max = int(np.floor(search_time_max / dt))
        self.step_ctr = 0

        self.plume_detected = False
        self.plume_detected_pos = None
        self.search_time = None

    def step(self):
        """
        Move the simulation forward one step.
        """
        self.step_ctr += 1
        self.agent.move(self.dt)

        if self.env.sample(self.agent.pos[0], self.agent.pos[1], dt=self.dt):
            self.plume_detected = True
            self.plume_detected_pos = self.agent.pos
            self.search_time = self.step_ctr * self.dt

    def run(self, with_plot=False, ax=None, draw_every=10, draw_background=True):
        """
        Step until plume is found, updating with plot if desired.

        :param with_plot: set to True to show plot
        :param ax: axis on which to draw plot
        :param draw_every: how many timesteps between plot updates
        :param draw_background: set to True to calculate and draw the background
            (you might want to set it to False if you're plotting multiple trajectories on one environment)
        """
        self.traj = []

        if with_plot:
            if draw_background:
                # show plume profiles
                heatmap, extent = self.env.heatmap()
                ax.matshow(heatmap.T, origin='lower', cmap=cm.hot, extent=extent, zorder=0)
                # show insect boundary
                bound = self.env.agent_search_radius
                kwargs = {'color': 'w', 'lw': 2}
                ax.vlines(-bound, ymin=-bound, ymax=bound, **kwargs)
                ax.vlines(bound, ymin=-bound, ymax=bound, **kwargs)
                ax.hlines(-bound, xmin=-bound, xmax=bound, **kwargs)
                ax.hlines(bound, xmin=-bound, xmax=bound, **kwargs)

                ax.set_xlim(extent[:2])
                ax.set_ylim(extent[2:])

            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            plt.draw()

        for step_ctr in range(self.n_steps_max):
            self.step()
            self.traj += [self.agent.pos.copy()]

            # update plot if it's time to do so
            if with_plot and (step_ctr % draw_every == 0 or step_ctr == self.n_steps_max - 1):

                ax.plot(np.array(self.traj)[:, 0], np.array(self.traj)[:, 1], c='b', lw=2, zorder=5)
                plt.draw()

            if self.plume_detected:
                if with_plot:
                    ax.plot(np.array(self.traj)[:, 0], np.array(self.traj)[:, 1], c='b', lw=2, zorder=5)
                    ax.scatter(self.agent.pos[0], self.agent.pos[1], marker='x', s=50, lw=4, c='c', zorder=10)
                    plt.draw()
                break


class Simulation(object):

    def __init__(self, plume_structure, agents, n_environments):
        pass