import os
import pickle
import sys

from matplotlib import pyplot as plt
from common.utils import get_date
import pprint
from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from typing import List
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBSplineBasisFunction
import random
import numpy as np
from basis_functions.wavelet.bspline.awr.awr_bspline_basisset import AWRBsplineBasisSet
from learners.sarsa_lambda import SarsaLambda
import gym
from common.runner import Runner, parallel_runs
from basis_functions.wavelet.bspline.fixed_bspline_basisset import FixedBSplineBasisSet

"""
This runs the code for AWR.
"""

def perform_experiments(env_name = 'MountainCar-v0', 
                        alpha: float = 0.015, 
                        tolerance: float = 0.5,
                        scale: int = 1,
                        do_proper = False,
                        num_runs=None
                        ):
    dic = {}
    env = gym.make(env_name)
    if num_runs is None or do_proper:
        num_runs = 14 if not do_proper else 100
    num_dims = env.observation_space.shape[0] if env_name != 'Acrobot-v1' else 4 
    def get_basis():
        return AWRBsplineBasisSet(num_observations=num_dims, scale=scale, order=2, awr_split_tolerance=tolerance)
    basis = get_basis()
    
    get_learner = lambda env: SarsaLambda([
            get_basis()  for _ in range(env.action_space.n)
        ], env.action_space.n, alpha=alpha, should_init_randomly=False, num_adaptive_episodes=10, gamma=1.0, lambd=0.9, epsilon=0.0)
    
    vals = parallel_runs(env_name, get_learner, num_runs=num_runs, verbose=1)
    L = get_learner(env)
    s = L.__repr__()
    print("Using params: ", dict(
        env_name=env_name,
        alpha=alpha,
        tolerance=tolerance,
        scale=scale,
        do_proper=do_proper,
    ))
    print("At the start we have: ", list(map(lambda x: len(x.functions), get_learner(env).basissets)))

    dic[basis.__repr__()] = list(vals) + [s, L.params()]

        

    dir = f'results/v2/main/{alpha}_{tolerance}_{scale}/{env_name}'
    
    if do_proper:
        dir = f'results/v2/proper_runs/{get_date()}/{env_name}'
        
    os.makedirs(dir, exist_ok=True)
    file = os.path.join(dir, 'results.p')
    with open(file, 'wb+') as f:
        pickle.dump(dic, f)
    
    
    for key, values in dic.items():
        print(f"Running {key}")
        mean, std, _, *_ = values
        x = np.arange(len(mean))
        plt.plot(x, mean, label=key)
        print(f"Mean for {key} = {np.mean(mean[-100:])} +- {np.mean(std[-100:])}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.5)
    plt.title(f"Basis for 400 episodes, averaged over num runs = {num_runs}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(dir, 'plot.png'))


if __name__ == '__main__':
    # Args: Env Name alpha, awr_tolerance, scale, do_proper
    if len(sys.argv) >= 5:
        do_proper = int(sys.argv.pop(-1))
    else:
        do_proper = False
    env_name = sys.argv.pop(1)
    alpha, tol, scale = sys.argv[1:]
    alpha = float(alpha)
    tol = float(tol)
    scale = int(scale)
    perform_experiments(env_name, alpha=alpha, tolerance=tol, scale=scale, do_proper=do_proper)
