# sarsa_lambda_linear.py - Sarsa Lambda with linear FA.
#
# Author: George Konidaris (gdk@cs.brown.edu)
# Creation Date: February 25th 2021 
# Copyright 2021 Brown University
#
# Assumptions:
#  - Discrete actions
#  - One identical FA per action
#  - Epsilon-greedy action selection
#  - Linear FA
#
# By default, the learning rate parameter is modified using Dabney
# and Barto's alpha bounds algorithm:
#  W. Dabney and A.G. Barto. Adaptive Step-Size for Online Temporal 
#  Difference Learning. Proceedings of the AAAI Conference on 
#  Artificial Intelligence, pages 872-878, 2012.
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

import numpy as np
import copy 
import random
import math 
import sys

class SarsaLambdaLinearFA:
    """
    Sarsa Lambda with linear function approximation.

    Methods
    -------
    clear_traces()
        zeros the trace vector (used at end of episode).
    zero_epsilon():
        sets the policy to purely greedy
    next_move(state)
        returns an epsilon-greedy action from the current state.
    Q(state, action):
        returns the Q function evaluated at a given state and action.
    max_Q(state)
        returns a tuple (value, a) giving the highest-valed action
        and its value, at a given state.
    update(s, a, r, sp, ap, terminal)
        runs a Sarsa update; second action and terminal flag optional.
    update_alpha(fa_vals, traces, fa_vals_p, traces_p, terminal)
        runs Dabney's alpha-bounds algorithm after an update (internal).
    

    Attributes (should not typically be accessed directly)
    ----------
    gamma : double
        discount factor, in [0, 1]
    lamb : double 
        eligibility trace parameter, in [0, 1], typically near 1.
    epsilon : double
        exploration rate, in [0,1], typically near 0.
    alpha : double
        learning rate / gradient descent term, in [0, 1], typically
        near 0.1 or lower.
    alpha_bounds : boolean
        indicates whether or not to use Dabney's alpha-bounds algorithm.
    weights : 2D numpy array 
        weight vector, indexed by basis function index and action.
    lambda_weights : 2D numpy array
        eligibility traces, one per weight. 
    n_actions : int
        the number of available actions
    fa : listo f linear function approximators
        one linear FA object per action.

    """
    def __init__(self, function_approximator, n_actions = None, alpha = 0.1, gamma=0.99, lamb=0.9, epsilon=0.05, initial_v=0.0, alpha_bounds=True):
        """
        Constructs a Sarsa Lambda instantiation. 

        Almost all parameters are optional, and if omitted are set to 
        something reasonable by default. 

        Parameters
        ----------
        function_approximator : a linear function approximator.
            either a list or a single instance; if a list, length
            should match n_actions. Deep copies are made and stored
            internally, so reference can be reused.
        n_actions : int, optional
            the dimensionality of the state space; if None, obtained by
            len(function_approximator) (which must in that case be a 
            list).
        alpha: double, optional
            learning rate (or gradient descent update rate), in [0,1].
            typically smaller than 0.1.
        gamma: double, optional
            discount factor in [0, 1], typically close to 1.
        lamb: double, optional
            eligibility trace parameter, in [0, 1], typically near 1.
        epsilon: double, optional
            exploration rate, in [0,1], typically near 0.
        initial_v: double, optional
            the value with which to initialize the value function.
        alpha_bounds: boolean, default.
            determines whether or not to use Dabney's alpha scaling
            algorithm, in which case the alpha parameter is simply
            the initialization.
        """     

        # Instance variables 
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.alpha = alpha 
        self.alpha_bounds = alpha_bounds 

        self.neginf = float("-inf")

        if(type(function_approximator) in [list,tuple,np.ndarray]):
            # There is one FA per action (check this)
            self.n_actions = len(function_approximator)
            if(n_actions is not None):
                if(n_actions != len(function_approximator)):
                    print("SarsaLambdaLinearFA warning: if `function_approximator' is a list it must have `n_actions' items. Overriding `n_actions'.", file = sys.stderr)

            # Copy the whole list
            self.fa = copy.deepcopy(function_approximator)
        else:
            # Create a new copy of the function approximator for each action.
            self.n_actions = n_actions
            if(n_actions is None):
                print("SarsaLambdaLinearFA error: if `function_approximator' is not a list, must provide `n_actions'.", file = sys.stderr)
                return
            self.fa = []
            for _ in range(0, n_actions):
                self.fa.append(copy.deepcopy(function_approximator))
        
        self.weights = np.zeros([self.fa[0].length(), n_actions])
        self.lambda_weights = np.zeros(self.weights.shape)

        # Initialize: assume zeroth bf is constant
        self.weights[0, :] = initial_v

    def clear_traces(self):
        """Clears the eligibility trace vector. Should be called
            at end of episode. 
        """

        self.lambda_weights = np.zeros(self.weights.shape)

    def zero_epsilon(self):
        """Makes the policy greedy."""

        self.epsilon = 0

    def next_move(self, state):
        """Stochastically eturns an epsilon-greedy action from the 
            current state. 
        """

        if(random.random() <= self.epsilon):
            return random.randrange(0, self.n_actions)

        # Build a list of actions evaluating to max_a Q(s, a)
        best = float("-inf")
        best_actions = []

        for a in range(0, self.n_actions):
            thisq = self.Q(state, a)

            if(math.isclose(thisq, best)):
                best_actions.append(a)
            elif(thisq > best):
                best = thisq
                best_actions = [a]                

        if((len(best_actions) == 0) or (math.isinf(best))):
            print("SarsaLambdaLinearFA: function approximator has diverged to infinity.", file = sys.stderr)
            return random.randrange(0, self.n_actions)

        # Select randomly among best-valued actions
        return random.choice(best_actions) 

    def Q(self, state, action):
        """Returns the Q value of the given state-action pair."""

        return np.dot(self.weights[:, action], self.fa[action].evaluate(state))

    def max_Q(self, state):
        """Returns the Q value of a given state, maximizing over 
            actions.
        """

        best = self.neginf
        for a in range(0, self.n_actions):
            qval = self.Q(state, a)
            if(qval > best):
                best = qval
                best_a = a
        return (best, best_a)

    def update(self, s, a, r, sp, ap=None, terminal=False):
        """
        Runs a Sarsa update, given a transition

        Parameters
        ----------
        s: numpy array
            the state at time t.
        a: int
            the action executed at time t.
        r: double
            the reward at time t.
        sp: numpy array
            the state at time t+1.
        ap: int, optional
            the action at time t+1. If not given, the algorithm picks
            the action with the maximum Q value.
        terminal: boolean, optional
            indicates that the transition has entered a terminal state.

        Returns
        -------
        double
            The computed TD error.  
        """

        # Compute TD error
        delta = r - self.Q(s, a)

        # Only include s' if it is not a terminal state.
        if(not terminal):
            if(ap is not None):
                delta += self.gamma*self.Q(sp, ap)
            else:
                (qp, ap) = self.max_Q(sp)
                delta += self.gamma*self.max_Q(qp)

        # Compute the basis functions for state s, action a.
        eval_fa = self.fa[a].evaluate(s)
        eval_fap = self.fa[ap].evaluate(sp)

        for each_a in range(0, self.n_actions):

            # Update traces
            self.lambda_weights[:, each_a] *= self.gamma*self.lamb
            if(each_a == a):
                self.lambda_weights[:, each_a] += eval_fa

            # Update weights
            self.weights[:, each_a] += self.alpha * delta * np.multiply(self.fa[each_a].get_gradient_factors(), self.lambda_weights[:, each_a])   

        # Run the alpha-bound algorithm if selected.
        if(self.alpha_bounds):
            self.update_alpha(eval_fa, self.lambda_weights[:, a], eval_fap, self.lambda_weights[:, ap], terminal)

        # Return the TD error, which may be informative. 
        return delta

    def update_alpha(self, fa_vals, traces, fa_vals_p, traces_p, terminal):
        """Runs Dabney's alpha-bounds algorithm after an update.
            This is really an internal function, and should not be
            called from the outside.
        """

        eps_alpha = 0.0

        eps_alpha = -1.0*np.dot(fa_vals, traces)
        if(not terminal):
            eps_alpha += self.gamma*np.dot(fa_vals_p, traces_p)
        
        if(eps_alpha < 0.0):
            self.alpha = min(math.fabs(-1.0 / eps_alpha), self.alpha) 

