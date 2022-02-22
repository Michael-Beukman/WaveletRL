from time import time
import os
import pickle
import sys

from matplotlib import pyplot as plt
from common.utils import get_date, save_compressed_pickle
import pprint
from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from typing import List
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBSplineBasisFunction
import random
import numpy as np
from basis_functions.wavelet.bspline.mawb.mawb_bspline_basisset import MAWBSplineBasisSet
from learners.sarsa_lambda import SarsaLambda
import gym
from common.runner import Runner, parallel_runs
from basis_functions.wavelet.bspline.fixed_bspline_basisset import FixedBSplineBasisSet

"""
This runs the code for MAWB Wavelet method, which basically combines both AWR and IBFDD
"""

def perform_experiments(env_name = 'MountainCar-v0', 
                        alpha: float = 0.015, 
                        awr_tolerance: float = 0.5,
                        ibfdd_tolerance: float = 0.5,
                        scale: int = 1,
                        num_adaptive_steps=10,
                        order: int = 2,
                        do_proper = False
                        ):
    dic = {}
    env = gym.make(env_name)
    num_runs = 14 if not do_proper else 100
    num_dims = env.observation_space.shape[0] if env_name != 'Acrobot-v1' else 4 
    def get_basis():
        return MAWBSplineBasisSet(num_observations=num_dims, scale=scale, order=order, awr_split_tolerance=awr_tolerance, ibfdd_add_tolerance=ibfdd_tolerance)
    basis = get_basis()
    
    get_learner = lambda env: SarsaLambda([
            get_basis()  for _ in range(env.action_space.n)
        ], env.action_space.n, alpha=alpha, should_init_randomly=False, num_adaptive_episodes=num_adaptive_steps, gamma=1.0, lambd=0.9, epsilon=0.0)
    Ps = dict(
        env_name=env_name,
        alpha=alpha,
        awr_tolerance=awr_tolerance,
        ibfdd_tolerance=ibfdd_tolerance,
        scale=scale,
        num_adaptive_steps=num_adaptive_steps,
        do_proper=do_proper,
    )
    dir = f'results/v4/gridsearch/{alpha}_{awr_tolerance}_{ibfdd_tolerance}_{scale}_{num_adaptive_steps}/{env_name}'
    if do_proper:
        dir = f'results/v4/proper_runs/{get_date()}/{env_name}'
    print("Using params: ", Ps)
    print("Saving to ", dir)
    vals = parallel_runs(env_name, get_learner, num_runs=num_runs, verbose=1, return_learners=True)
    L = get_learner(env)
    s = L.__repr__()
    print("Using params: ", Ps)
    print("At the start we have: ", list(map(lambda x: len(x.functions), get_learner(env).basissets)))

    dic[basis.__repr__()] = list(vals) + [s, L.params()]

    os.makedirs(dir, exist_ok=True)
    file = os.path.join(dir, 'results.p')
    save_compressed_pickle(file, dic)
    # with open(file, 'wb+') as f:
    #     pickle.dump(dic, f)
    
    
    for key, values in dic.items():
        print(f"Running {key}")
        mean, std, _, learners, *_ = values

        x = np.arange(len(mean))
        plt.plot(x, mean, label=key)
        print(f"Mean for {key} = {np.mean(mean[-100:])} +- {np.round(np.mean(std[-100:]))}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.5)
    plt.title(f"Basis for 400 episodes, averaged over num runs = {num_runs}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(os.path.join(dir, 'plot.png'))

if __name__ == '__main__':
    # Args: Env Name alpha, awr_tolerance ibfdd_tolerance, scale, number_of_adaptive_episodes do_proper
    env_name = sys.argv.pop(1)
    alpha, tol_awr, tol_ibfdd, scale, num_adap_eps, do_proper = sys.argv[1:]
    alpha = float(alpha)
    tol_awr = float(tol_awr)
    tol_ibfdd = float(tol_ibfdd)
    scale = int(scale)
    num_adap_eps = int(num_adap_eps)
    do_proper = int(do_proper)
    t = time()
    perform_experiments(env_name, 
                        alpha=alpha, 
                        awr_tolerance=tol_awr, 
                        ibfdd_tolerance=tol_ibfdd,
                        num_adaptive_steps=num_adap_eps,
                        scale=scale, do_proper=do_proper)
    e = time()
    print(f"This run took {e - t} seconds")