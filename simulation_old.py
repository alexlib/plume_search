from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm


class Simulation(object):
    """
    Class for running a simulation of an agent moving through an environment
    filled with plume sources.
    :param hit_probability_function: function calculating hit probability at an array of displacements from a source
    :param params: parameter dict for hit_probability function
    :param src_density: density of sources (#/m^2)
    :param search_time_max: maximum search time (s)
    :param dt: timestep
    :param plume_bdry_hit_prob: value that hit probability must be lower than to be considered outside the plume
    :param plume_map_resolution: resolution (num_pix_x, num_pix_y) to use when drawing plume_map if plotting is desired
    """

    def __init__(self, hit_probability_function, params,
                 src_density, search_time_max, dt, plume_bdry_hit_prob=1e-3,
                 plume_map_resolution=(100, 100)):

        self._agent = None
        self.hit_probability_function = hit_probability_function
        self.params_hpf = params
        self.src_density = src_density
        self.search_time_max = search_time_max
        self.dt = dt
        self.plume_bdry_hit_prob = plume_bdry_hit_prob
        self.plume_map_resolution = plume_map_resolution
        self.n_steps_max = int(np.ceil(search_time_max / dt))

        def hit_prob_short(dx, dy):
            return self.hit_probability_function(dx, dy, **self.params_hpf)
        self.hit_prob_short = hit_prob_short

        # set null hidden variable that will contain displayable map
        self._plume_map = None

        # set other variables to null values
        self.n_srcs = None
        self.src_positions = None
        self.plume_found = False
        self.search_time = None
        self.pos_plume_found = None
        self.step_ctr = 0

        self.traj = None

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, agent):
        """
        :param agent: tracking agent instance
        """
        self._agent = agent
        # determine effective plume boundaries
        bdry_plume = [0, 0, 0, 0]  # [x_min, x_max, y_min, y_max]
        directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

        for ctr, direction in enumerate(directions):
            # move in this direction until the hit probability becomes less than bdry_prob
            if direction[-1] == 0:
                dr = np.array([0, 0], dtype=float)
            else:
                dr = np.array([bdry_plume[1]/2, 0])
            keep_going = True
            while keep_going:
                hit_prob = self.hit_prob_short(*dr)
                if hit_prob < self.plume_bdry_hit_prob:
                    if ctr in [0, 1]:
                        bdry_plume[ctr] = dr[0]
                    elif ctr in [2, 3]:
                        bdry_plume[ctr] = dr[1]
                    keep_going = False
                else:
                    dr += direction * self._agent.speed * self.dt

        self.bdry_plume = np.array(bdry_plume)

        # determine agent boundary
        dist_max = self.search_time_max * self._agent.speed
        bdry_agent = [-dist_max, dist_max, -dist_max, dist_max]  # [x_min, x_max, y_min, y_max]
        self.bdry_agent = np.array(bdry_agent)

        # determine environment boundary (region in which sources can be distributed)
        self.bdry_env = [0, 0, 0, 0]
        self.bdry_env[0] = self.bdry_agent[0] - self.bdry_plume[1]
        self.bdry_env[1] = self.bdry_agent[1] - self.bdry_plume[0]
        self.bdry_env[2] = self.bdry_agent[2] - self.bdry_plume[3]
        self.bdry_env[3] = self.bdry_agent[3] - self.bdry_plume[2]

        self.bdry_env = [-30, 10, -15, 15]
        self.area_env = (self.bdry_env[1] - self.bdry_env[0]) * (self.bdry_env[3] - self.bdry_env[2])


    def reset(self):
        self.plume_found = False
        self.search_time = None
        self.pos_plume_found = None
        self.step_ctr = 0

    def set_src_positions(self, src_positions):
        """
        Set all source positions in the environment.
        :param src_positions: list/array of source positions
        """
        if isinstance(src_positions, str):
            if src_positions == 'random':
                # sample number of sources from Poisson distribution
                self.n_srcs = np.random.poisson(self.area_env * self.src_density)
                # get src positions
                pos_min = [self.bdry_env[0], self.bdry_env[2]]
                pos_max = [self.bdry_env[1], self.bdry_env[3]]
                self.src_positions = np.random.uniform(pos_min, pos_max, (self.n_srcs, 2))

        else:
            self.n_srcs = len(src_positions)
            self.src_positions = src_positions

        self._plume_map = None

    def step(self):
        """
        Move the simulation forward one step.
        """
        self.step_ctr += 1
        self.agent.move(self.dt)
        if self.n_srcs:
            dspl_to_srcs = self.agent.pos - self.src_positions
            miss_prob = (1 - self.hit_prob_short(dspl_to_srcs[:, 0], dspl_to_srcs[:, 1])).prod()
        else:
            miss_prob = 1

        hit_prob = 1 - miss_prob

        #print(hit_prob)
        if self.agent.detect_odor(hit_prob):
            self.plume_found = True
            self.search_time = self.step_ctr * self.dt
            self.pos_plume_found = self.agent.pos

    def run(self, with_plot=False, ax=None, draw_every=10):
        """
        Step until plume is found, updating with plot if desired.
        :param with_plot: set to True to show plot
        :param ax: axis on which to draw plot
        :param draw_every: how many timesteps between plot updates
        """
        self.traj = []

        if with_plot:
            # show plume profiles
            ax.matshow(self.plume_map.T, origin='lower', cmap=cm.hot, extent=self.bdry_env, zorder=0)
            # show insect boundary
            ax.vlines(self.bdry_agent[0], ymin=self.bdry_agent[2], ymax=self.bdry_agent[3], color='w', lw=2)
            ax.vlines(self.bdry_agent[1], ymin=self.bdry_agent[2], ymax=self.bdry_agent[3], color='w', lw=2)
            ax.hlines(self.bdry_agent[2], xmin=self.bdry_agent[0], xmax=self.bdry_agent[1], color='w', lw=2)
            ax.hlines(self.bdry_agent[3], xmin=self.bdry_agent[0], xmax=self.bdry_agent[1], color='w', lw=2)

            ax.set_xlim(self.bdry_env[:2])
            ax.set_ylim(self.bdry_env[2:])

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

            if self.plume_found:
                if with_plot:
                    ax.plot(np.array(self.traj)[:, 0], np.array(self.traj)[:, 1], c='b', lw=2, zorder=5)
                    ax.scatter(self.agent.pos[0], self.agent.pos[1], marker='x', s=50, lw=4, c='c', zorder=10)
                    plt.draw()
                break

    @property
    def plume_map(self):
        if self._plume_map is None:
            bins_x = np.linspace(self.bdry_env[0], self.bdry_env[1], self.plume_map_resolution[0])
            bins_y = np.linspace(self.bdry_env[2], self.bdry_env[3], self.plume_map_resolution[1])

            x = 0.5 * (bins_x[:-1] + bins_x[1:])
            y = 0.5 * (bins_y[:-1] + bins_y[1:])

            xm, ym = np.meshgrid(x, y, indexing='ij')

            # calculate miss probability
            prob_miss = np.ones(xm.shape, dtype=float)

            for src_position in self.src_positions:
                dx = xm - src_position[0]
                dy = ym - src_position[1]
                prob_miss *= (1 - self.hit_prob_short(dx, dy))

            # calculate hit probability
            prob_hit = 1 - prob_miss
            self._plume_map = prob_hit

        return self._plume_map