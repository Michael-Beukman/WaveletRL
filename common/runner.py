import pickle
import random
from basis_functions.wavelet.bspline.fixed_bspline_basisset import FixedBSplineBasisSet
from matplotlib import pyplot as plt
from basis_functions.fourier.fourier_basis_set import FourierBasisSet
from typing import Callable, List
import gym
import numpy as np
from learners.sarsa_lambda import SarsaLambda
import ray
"""This file contains functions to run a set of reproducible experiments, in parallel.
"""


class Runner:
    def __init__(self, learner: SarsaLambda, env: gym.Env, is_acrobot: bool = False, verbose: int = 0, seed: int=0) -> None:
        self.env = env
        self.learner = learner
        self.is_acrobot = is_acrobot
        self.verbose = verbose
        self.interval = 100
        self.seed = seed
    
    def run(self, num_episodes: int = 400) -> List[float]:
        rewards = []
        for episode in range(1, num_episodes+1):
            reward_this_episode = 0
            self.learner.zero_trace()
            done = False
            if self.is_acrobot:
                self.env.reset()
                state = self.norm(self.env.state)
            else:
                state = self.norm(self.env.reset())

            action = self.learner.get_action(state)
            while not done:
                next_state, r, done, info = self.env.step(action)
                if self.is_acrobot:
                    next_state = self.norm(self.env.state)
                else:
                    next_state = self.norm(next_state)
                reward_this_episode += r

                next_action = self.learner.get_action(next_state)

                self.learner.add_sample(state, action, r, next_state, next_action, done and not (info.get('TimeLimit.truncated', False)))
                action = next_action
                state = next_state
            rewards.append(reward_this_episode)
            if episode % self.interval == 0 and self.verbose > 0:
                print(f"{self.seed}: Average Reward over {self.interval} episodes at episode {episode} = {np.mean(rewards[-self.interval:])}")
        return rewards


    def norm(self, obs: np.ndarray):
        # From George Konidaris' code.
        if self.is_acrobot:
            env = self.env
            max = np.array([np.pi, np.pi, env.observation_space.high[4], env.observation_space.high[5]])
            min = np.array([-np.pi, -np.pi, env.observation_space.low[4], env.observation_space.low[5]])
        else:
            min, max = self.env.observation_space.low, self.env.observation_space.high
        return (obs - min) / (max - min)

def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator
    
    
def parallel_runs(env_name: str, func_to_get_learner: Callable[[gym.Env], SarsaLambda], num_runs:int = 4, verbose: int = 0, return_learners=False, _episodes:int = 400):
    """Runs this in parallel

    Args:
        env_name (str): gym.make argument
        func_to_get_learner (Callable[[gym.Env], SarsaLambda]): Given a environment, return a learner
        num_runs (int, optional): How many runs (with different seeds) should be performed. Defaults to 4.
        verbose (int, optional): Controls verbosity of output. Defaults to 0.
        return_learners (bool, optional): If true, also returns the learners. Useful for interacting with the basis functions, etc. Defaults to False.
        _episodes (int, optional): How many episodes to run. Defaults to 400.

    Returns:
        mean, std, all_results, [learners], where:
            Mean is an (_episodes, ) shaped array containing the mean reward over runs per episode
            std: Same as mean, just standard deviation now.
            all_results: (num_runs, episodes) containing raw rewards for all runs
            [learners]: Only true if return_learners is, list of length num_runs containing the SarsaLambda objects.
    """
    @conditional_decorator(ray.remote, num_runs > 1)
    def single_run(seed: int):
        np.random.seed(seed)
        random.seed(seed)
        env = gym.make(env_name)
        env.seed(seed)

        learner = func_to_get_learner(env)
        runner = Runner(learner, env, is_acrobot='Acrobot-v1' in env_name, verbose=verbose, seed=seed)
        answers = runner.run(_episodes)
        try:
            print("At the end we have: ", list(map(lambda x: len(x.functions), learner.basissets)))
        except Exception as e:
            pass
        try:
            print("At the end we added: ", list(map(lambda x: f"{x.ibfdd_added} IBFDD Funcs & {x.awr_added} AWR Funcs", learner.basissets)))
        except Exception as e:
            pass
        if return_learners:
            return answers, learner
        return answers
    if num_runs > 1:
        ray.init()
    if num_runs > 1:
        futures = [single_run.remote(i) for i in range(num_runs)]
        all_results = ray.get(futures)
    else:
        all_results = [single_run(i) for i in range(num_runs)]
    
    if return_learners:
        learners = [r[1] for r in all_results]
        all_results = [r[0] for r in all_results]
    all_results = np.array(all_results)

    new_results = []
    for r in all_results:
        new_results.append([
            r[i] for i in range(0, len(r))
        ])
    
    new_results = np.array(new_results)
    
    # new_results = all_results
    mean, std = np.mean(new_results, axis=0), np.std(new_results, axis=0)
    x = np.arange(len(mean))
    ray.shutdown()
    if return_learners:
        return mean, std, all_results, learners
    return mean, std, all_results
