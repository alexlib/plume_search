from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from db_api import session, models

SIM_ID = 'search_linear_2D_gaussian_plumes_identical_vary_theta_r0.02_d0.02_w0.5_speed_0.5_density0.04_T20_N2000_hit_prob_bound0' \
         '.001'

# get simulation and list of all agents ordered by their theta
sim = session.query(models.Simulation).get(SIM_ID)

print('Simulation:')
print(sim)
print('Plume:')
print(sim.plume)

agents = session.query(models.Agent).filter(models.Agent.simulation == sim).\
    order_by(models.Agent.params.theta).all()

thetas = np.array([agent.params_linear.theta for agent in agents])
p_detect = np.array([agent.detection_probability for agent in agents])
confidence = np.array([agent.detection_probability_confidence for agent in agents])

fig, ax = plt.subplots(1, 1, facecolor='white')
ax.errorbar(thetas, p_detect, yerr=(p_detect - confidence[:, 0], p_detect + confidence[:, 1]), lw=2)

plt.show()