import os
import pickle
import sys

from matplotlib import pyplot as plt
import numpy as np
from common.utils import get_date
from basis_functions.fourier.fourier_basis_set import FourierBasisSet
import gym
from basis_functions.wavelet.bspline.fixed_bspline_basisset import FixedBSplineBasisSet
from learners.sarsa_lambda import SarsaLambda
from common.runner import parallel_runs
"""
This runs the code for the fixed basis sets.
"""

def perform_experiments(env_name = 'MountainCar-v0',
                        gamma=0.99, 
                        lambd=0.85, 
                        epsilon=0.05, 
                        fixed_alpha=0.015,
                        
                        do_proper_run = False,
                        ):
    """
    This runs a set of experiments of a given environment and some parameters. This file runs many different fixed basis sets, including multiple configurations of B-spline and multiple orders of the fourier basis.
    Args:
        env_name (str, optional): Environment to use. Defaults to 'MountainCar-v0'.
        gamma (float, optional): Reward discount factor. Defaults to 0.99.
        lambd (float, optional): Eligibility trace decay rate. Defaults to 0.85.
        epsilon (float, optional): Epsilon to use in a e-greedy policy. Defaults to 0.05.
        fixed_alpha (float, optional): The learning rate, can be a list of learning rates for each different function. Defaults to 0.015.
        do_proper_run (bool, optional): If True, runs 100 runs, otherwise only 14 -- to be used for e.g. a grid search. Defaults to False.

    """
    dic = {}
    env = gym.make(env_name)
    # Get the different basis functions
    def get_spline_10(dims):
        return FixedBSplineBasisSet(dims, order=1, scale=0)

    def get_spline_20(dims):
        return FixedBSplineBasisSet(dims, order=2, scale=0)

    def get_spline_21(dims):
        return FixedBSplineBasisSet(dims, order=2, scale=1)
    
    def get_spline_22(dims):
        return FixedBSplineBasisSet(dims, order=2, scale=2)
    
    def get_spline_42(dims):
        return FixedBSplineBasisSet(dims, order=4, scale=2)

    def get_fourier3(dims):
        return FourierBasisSet(dims, order=3)
    
    def get_fourier7(dims):
        return FourierBasisSet(dims, order=7)
    
    def get_fourier(dims):
        return FourierBasisSet(dims, order=5)

    funcs = [
        get_spline_10,
        get_spline_20,
        get_spline_21,
        get_spline_22,
        get_spline_42,
        
        get_fourier,
        get_fourier3,
        get_fourier7,
    ]
    params = dict(gamma=gamma, lambd=lambd, epsilon=epsilon)
    
    num_runs = 14 if not do_proper_run else 100
    if type(fixed_alpha) != list:
        fixed_alphas = [fixed_alpha] * len(funcs)
    else:
        fixed_alphas = fixed_alpha
    K = 0
    
    # Now, go over each method
    for f, fixed_alpha in zip(funcs, fixed_alphas):
        num_dims = env.observation_space.shape[0] if env_name != 'Acrobot-v1' else 4 
        tmp_f = f(num_dims)
        K += 1
        basis = f(num_dims)
        
        get_learner = lambda env: SarsaLambda([
               f(num_dims)  for _ in range(env.action_space.n)
            ], env.action_space.n, should_init_randomly=False, perform_alpha_scaling=False, alpha=fixed_alpha, **params)
        
        # Run in parallel
        vals = parallel_runs(env_name, get_learner, num_runs=num_runs, verbose=1)
        L = get_learner(env)
        s = L.__repr__()

        dic[basis.__repr__()] = list(vals) + [s, L.params()]

        # Test Dabney's Alpha scaling, which was empirically found to not perform very well.
        if isinstance(tmp_f, FourierBasisSet):
            get_learner = lambda env: SarsaLambda([
               f(num_dims)  for _ in range(env.action_space.n)
            ], env.action_space.n, alpha=1, 
                                    should_init_randomly=False, 
                                    perform_alpha_scaling=True, 
                                    **params)
            
            vals = parallel_runs(env_name, get_learner, num_runs=num_runs, verbose=1)
            L = get_learner(env)
            s = L.__repr__()

            dic[basis.__repr__() + "_Alpha Scale"] = list(vals) + [s, L.params()]

        
    # Plot and save results.
    dirname_to_store = f"{gamma}_{lambd}_{epsilon}_{fixed_alpha}"
    dir = f'results/v1/main/{dirname_to_store}/{env_name}'
    if do_proper_run:
        dir = f'results/v1/proper_runs/{get_date()}/{env_name}'
    os.makedirs(dir, exist_ok=True)
    file = os.path.join(dir, 'results.p')
    with open(file, 'wb+') as f:
        pickle.dump(dic, f)
    
    plt.figure(figsize=(20,20))
    for key, values in dic.items():
        print(f"Running {key}")
        mean, std, _, _, _ = values
        x = np.arange(len(mean))
        plt.plot(x, mean, label=key)
        plt.fill_between(x, mean - std, mean + std, alpha=0.3)
        print(f"Mean for {key} = {np.mean(mean[-100:])}")
    plt.title(f"Basis for 400 episodes, averaged over num runs = {num_runs}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.savefig(os.path.join(dir, 'plot.png'), dpi=400)


if __name__ == '__main__':
    # Parse command line arguments
    # Order: Env Name gamma lambd eps ...alphas do_proper
    if len(sys.argv) > 6:
        do_proper = int(sys.argv.pop(-1))
    else:
        do_proper = False
    env_name = sys.argv.pop(1)
    gamma, lambd, eps, *alphas = map(float, sys.argv[1:])
    if len(alphas) == 1: alphas = alphas[0]
    print("alpha = ", alphas)
    perform_experiments(env_name, gamma=gamma, lambd=lambd, epsilon=eps, fixed_alpha=alphas, do_proper_run=do_proper)