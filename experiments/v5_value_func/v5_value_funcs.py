import gym
from matplotlib import cm, pyplot as plt
import numpy as np
from learners.sarsa_lambda import SarsaLambda
import os
import pickle
import sys

import numpy as np
from common.utils import get_date, mysavefig
from basis_functions.fourier.fourier_basis_set import FourierBasisSet
import gym
from basis_functions.wavelet.bspline.fixed_bspline_basisset import FixedBSplineBasisSet
from common.runner import parallel_runs
from basis_functions.wavelet.bspline.ibfdd.ibfdd_bspline_basisset import IBFDDBSplineBasisSet
from common.utils import clean_label

"""
    This file basically trains using the different basis functions, and then plots their value functions.
"""


def learn_and_save(fixed_alpha, episodes=400, env_name: str = 'MountainCar-v0', gamma=1, lambd=0.9, epsilon=0,):
    dic = {}
    env = gym.make(env_name)

    def get_spline_10(dims):
        return FixedBSplineBasisSet(dims, order=1, scale=0)

    def get_spline_20(dims):
        return FixedBSplineBasisSet(dims, order=2, scale=0)

    def get_spline_21(dims):
        return FixedBSplineBasisSet(dims, order=2, scale=1)
    
    def get_spline_22(dims):
        return FixedBSplineBasisSet(dims, order=2, scale=2)
    
    def get_spline_32(dims):
        return FixedBSplineBasisSet(dims, order=3, scale=2)
    
    
    def get_spline_42(dims):
        return FixedBSplineBasisSet(dims, order=4, scale=2)

    def get_fourier3(dims):
        return FourierBasisSet(dims, order=3)
    
    def get_fourier7(dims):
        return FourierBasisSet(dims, order=7)
    
    def get_fourier(dims):
        return FourierBasisSet(dims, order=5)

    def get_ibfdd(dims):
        return IBFDDBSplineBasisSet(num_observations=dims, scale=1, order=2, awr_split_tolerance=1, ibfdd_add_tolerance=5)
    
    funcs = [
        get_ibfdd, 
        get_spline_10,
        get_spline_20,
        get_spline_21,
        get_spline_22,
        get_spline_32,
        get_spline_42,
        
        get_fourier3,
        get_fourier,
        get_fourier7,
    ]
    params = dict(gamma=gamma, lambd=lambd, epsilon=epsilon)
    
    num_runs = 1
    
    if type(fixed_alpha) != list:
        fixed_alphas = [fixed_alpha] * len(funcs)
    else:
        fixed_alphas = fixed_alpha
    assert len(fixed_alphas) == len(funcs)
    print("ALPHA = ", fixed_alphas)
    K = 0
    for f, fixed_alpha in zip(funcs, fixed_alphas):
        num_dims = env.observation_space.shape[0] if env_name != 'Acrobot-v1' else 4 
        K += 1
        basis = f(num_dims)
        
        get_learner = lambda env: SarsaLambda([
               f(num_dims)  for _ in range(env.action_space.n)
            ], env.action_space.n, should_init_randomly=False, perform_alpha_scaling=False, alpha=fixed_alpha, **params)
        
        vals = parallel_runs(env_name, get_learner, num_runs=num_runs, verbose=1, return_learners=True, _episodes=episodes)
        L = get_learner(env)
        s = L.__repr__()

        dic[basis.__repr__()] = list(vals) + [s, L.params()]
        # break

    dirname_to_store = f"{gamma}_{lambd}_{epsilon}_{fixed_alpha}_{episodes}_{len(funcs)}"
    dir = f'results/v5_learners/{dirname_to_store}/{env_name}'
    os.makedirs(dir, exist_ok=True)
    file = os.path.join(dir, 'results.p')
    with open(file, 'wb+') as f:
        pickle.dump(dic, f)

def get_values(learner: SarsaLambda):
    N = 500
    x1samp = np.linspace(0.0, 1.00, N)
    x2samp = np.linspace(0.0, 1.00, N)

    xs, ys = np.meshgrid(x1samp, x2samp)
    zs = np.zeros(xs.shape)

    for i in range(0, zs.shape[0]):
        for j in range(0, zs.shape[1]):
            s = [xs[i, j], ys[i, j]]
            features = [
                bset.get_value(s) for bset in learner.basissets
            ]
            zq = learner.V(features)[0]
            zs[i, j] = -1.0*zq

    return xs, ys, zs


def plot_value_func(learner: SarsaLambda):
    fig = plt.figure(constrained_layout=True)
    ax = fig.gca(projection='3d')
    xs, ys, zs = get_values(learner)
    plt.xlabel("$x$")
    plt.ylabel("$\dot x$")
    ax.plot_surface(xs, ys, zs, cmap=cm.get_cmap("coolwarm"),
                       linewidth=0, antialiased=False)
    ax.view_init(elev=45, azim=45)

def plot_value_func_contour(learner: SarsaLambda):
    xs, ys, zs = get_values(learner)
    plt.xlabel("$x$")
    plt.ylabel("$\dot x$")
    plt.contourf(xs, ys, zs, levels=50, cmap='inferno')



def evaluate_learner(learner: SarsaLambda, env_name='MountainCar-v0'):
    learner.alpha = 0
    learner.epsilon = 0
    learner.num_adaptive_episodes = -1
    import copy
    def f(x):
        L = copy.deepcopy(learner)
        return L
    vals = parallel_runs(env_name, f, num_runs=10, verbose=1, return_learners=True, _episodes=100)
    return vals

def do_all_plots(func_to_plot=plot_value_func, name=''):
    F = 'results/v5_learners/1_0.9_0_0.003_10000_9/MountainCar-v0/results.p'
    D = '/'.join(F.split("/")[:-1]) + f"/value_funcs{name}"
    os.makedirs(D, exist_ok=True)
    with open(F, 'rb') as f:
        dic = pickle.load(f)
    keys = list(dic.keys())
    # K = keys[-3]
    K = keys[0]
    for K in keys:
        print(K)
        vals = dic[K]
        L = vals[3]
        assert len(L) == 1
        learner: SarsaLambda = L[0]
        func_to_plot(learner)
        
        vals = evaluate_learner(learner)
        mean_rew = np.mean(vals[0][-100:])
        alls = vals[2]
        assert alls.shape == (10, 100)
        V = np.mean(alls, axis=1)
        Mean = np.mean(V)
        std = np.std(V)

        mean_rew = Mean
        clean_K = K.replace("(", "").replace(")", "").replace("=", "").replace(" ","_")
        K = clean_label(K)
        if "IBFDD" in K:
            K = K.replace("IBFDD", "Decoupled B-Spline")
            K = K.replace(", ibfdd_add_tolerance=5", "")
        plt.title(K + "\nMean Reward over 100 steps = " + str(int(np.round(mean_rew))) + f" ({np.round(std)})")
        mysavefig(os.path.join(D, clean_K), pad_inches=0, bbox_inches='tight')
        plt.close()
    

if __name__ == '__main__':
    do_all_plots(plot_value_func);
    do_all_plots(plot_value_func_contour, '_contour');
    # Run this to train the learners
    # learn_and_save([0.005, 0.15, 0.003, 0.015, 0.0015, 0.001, 0.0015, 0.0015, 0.003], episodes=10000)