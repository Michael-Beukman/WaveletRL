# acro_sarsa_lambda.py - Acrobat with Fourier BasisSarsa Lambda.
#
# Author: George Konidaris (gdk@cs.brown.edu)
# Creation Date: February 26th 2021 
# Copyright 2021 Brwown University
#
# NOTE: for some weird reason, the state returned by the Acrobot-v1 
# Gym environment comes pre-expanded with a sort of hacky
# first-order FB - but only the angles, not the velocities. I have
# no idea why one would do that, so the code below manually hacks 
# the state to match the original formulation. 
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
import fourier_fa
from sarsa_lambda_linear import SarsaLambdaLinearFA
from fourier_fa import FourierBasis
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

if __name__ == "__main__":

    samples = 5
    episodes = 100
    gamma = 1.0
    order = 7     
    zero_epsilon_last_episodes = 5
    run_data = np.zeros([samples, episodes])

    env = gym.make('Acrobot-v1')

    # Work around the weird sin/cosine stuff. Why dudes?? Not in the original.
    state_dim = 4
    n_actions = env.action_space.n
    u_state = np.array([np.pi, np.pi, env.observation_space.high[4], env.observation_space.high[5]])
    l_state = np.array([-np.pi, -np.pi, env.observation_space.low[4], env.observation_space.low[5]])
    d_state = u_state - l_state

    for sample in range(0, samples):

        fb = FourierBasis(order=order, d=state_dim)
        learner = SarsaLambdaLinearFA(fb, n_actions = n_actions, gamma=gamma, lamb=0.90, epsilon=0.05, alpha_bounds=True)

        for episode in range(0, episodes):

            if(episode >= episodes - zero_epsilon_last_episodes):
                learner.zero_epsilon()

            learner.clear_traces()
            env.reset()
            # How many mutha'uckas?
            s = env.state 
            s = (s - l_state) / d_state 
            a = learner.next_move(s) 

            done = False
            nsteps = 0
            sum_r = 0.0

            while(not done):
                _, r, done, info = env.step(a) 
                # Too many to count, mutha'uckas.
                sp = env.state 
                sp = (sp - l_state) / d_state 

                term = (done and not (info.get('TimeLimit.truncated', False)))
                ap = learner.next_move(sp) 

                learner.update(s, a, r, sp, ap, terminal=term) 
                s = sp
                a = ap
                sum_r += r*pow(gamma, nsteps)
                nsteps += 1

            run_data[sample, episode] = sum_r
            print(str(sample + 1) + ", " + str(episode + 1) + ": " + str(sum_r)) 

        env.close()
    
    data_mean = np.mean(run_data, axis=0)
    data_std = np.std(run_data, axis=0) 
    data_range = range(0, episodes)


    plt.plot(data_range, data_mean)
    plt.fill_between(data_range, data_mean-data_std, data_mean+data_std, alpha=0.5)
    plt.title("Acrobot FB order " + str(fb.order))
    plt.show()