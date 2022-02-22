# mc_sarsa_lambda.py - Mountain Car with Fourier BasisSarsa Lambda.
#
# Author: George Konidaris (gdk@cs.brown.edu)
# Creation Date: February 25th 2021 
# Copyright 2021 Brown University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import gym
from sarsa_lambda_linear import SarsaLambdaLinearFA
from fourier_fa import FourierBasis
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

if __name__ == "__main__":

    samples = 10
    episodes = 100
    gamma = 1.0
    order = 8
    zero_epsilon_last_episodes = 10 
    run_data = np.zeros([samples, episodes])

    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    u_state = env.observation_space.high
    l_state = env.observation_space.low
    d_state = u_state - l_state

    for sample in range(0, samples):

        fb = FourierBasis(order=order, d=state_dim)
        learner = SarsaLambdaLinearFA(fb, n_actions = n_actions, gamma=gamma, lamb=0.95, epsilon=0.05)

        for episode in range(0, episodes):

            if(episode >= episodes - zero_epsilon_last_episodes):
                learner.zero_epsilon()

            learner.clear_traces()
            s = (env.reset() - l_state) / d_state 
            a = learner.next_move(s) 

            done = False
            nsteps = 0
            sum_r = 0.0

            while(not done):
                sp, r, done, info = env.step(a) 
                sp = (sp - l_state) / d_state 
                term = (done and not (info.get('TimeLimit.truncated', False)))
                ap = learner.next_move(sp) 

                learner.update(s, a, r, sp, ap, terminal=term) 
                s = sp
                a = ap
                sum_r += r*pow(gamma, nsteps)
                nsteps += 1

            run_data[sample, episode] = sum_r
            print(str(sample + 1) + ", " +  str(episode + 1) + ": " + str(sum_r)) 

        env.close()
    
    data_mean = np.mean(run_data, axis=0)
    data_std = np.std(run_data, axis=0) 
    data_range = range(0, episodes)


    plt.plot(data_range, data_mean)
    plt.fill_between(data_range, data_mean-data_std, data_mean+data_std, alpha=0.5)
    plt.title("MountainCar FB order " + str(fb.order))
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x1samp = np.linspace(0.0, 1.00, 100)
    x2samp = np.linspace(0.0, 1.00, 100)

    xs, ys = np.meshgrid(x1samp, x2samp)
    zs = np.zeros(xs.shape)

    for i in range(0, zs.shape[0]):
        for j in range(0, zs.shape[1]):
            s = [xs[i, j], ys[i, j]]
            (zq, _) = learner.max_Q(s)
            zs[i, j] = -1.0*zq
    ax.plot_surface(xs, ys, zs, cmap=cm.get_cmap("coolwarm"),
                       linewidth=0, antialiased=False)
    ax.view_init(elev=45, azim=45)
    plt.show()