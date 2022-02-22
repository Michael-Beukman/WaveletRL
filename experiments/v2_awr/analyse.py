from collections import defaultdict
import glob
import os
import pickle
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
from common.utils import get_date, load_compressed_pickle, mysavefig
import numpy as np
import seaborn as sns
from common.utils import clean_label
sns.set_theme()

def analyse_main(name='MC'):
    """This basically takes some pickle files from all of the runs, and plots the results to a shared figure in results/v2
    """
    def is_key_good(key):
        L = key.lower()
        if 'mawb' in L or 'ibfdd' in L or 'awr' in L: return True
        if 'spline' in key.lower():
            return 'scale=2' in L and 'order=2' in L
        else:
            return 'order=5' in key and 'Alpha' not in key

    # The files to use
    if name == 'MC':
        Fs = [
        'results/v1/proper_runs/2022-02-12_15-25-50/MountainCar-v0/results.p',
        'results/v2/proper_runs/2022-02-12_19-11-23/MountainCar-v0/results.p',
        'results/v3/proper_runs/2022-02-19_08-46-02/MountainCar-v0/results.p',
        'results/v4/proper_runs/2022-02-19_15-18-23/MountainCar-v0/results.p'
        ]
    else:
    # acrobot
        Fs = [
            'results/v1/proper_runs/2022-02-15_08-46-12/Acrobot-v1/results.p',
            'results/v2/proper_runs/2022-02-16_14-15-36/Acrobot-v1/results.p',
            'results/v3/proper_runs/2022-02-19_08-08-56/Acrobot-v1/results.p',
            'results/v4/proper_runs/2022-02-21_09-26-25/Acrobot-v1/results.p.pbz2'
        ]
    dic = {}
    for f2 in Fs:
        if 'pbz2' in f2:
            dic.update(load_compressed_pickle(f2))
        else:
            with open(f2, 'rb') as f:
                dic.update(pickle.load(f))
    d = Fs[1]

    plt.figure(figsize=(10,10))
    # Go over each and plot them
    for key, values in dic.items():
        if not is_key_good(key): continue
        mean, std, alls, *names = values
        
        # Average over last 20 episodes
        W = 19
        new_alls = np.zeros_like(alls)
        for i, run in enumerate(alls):
            for j in range(len(alls[i])):
                new_alls[i][j] = np.mean(alls[i][max(j-W, 0):j+1])
        
        alls = new_alls
        mean, std = np.mean(alls, axis=0), np.std(alls, axis=0)
        if len(names) >= 2:
            params = names[-1]
            alph = params['alpha']
            print('PARAMS', params)
        else:
            # Did not save this in the params, but the name still contains the alpha
            assert "alpha=0.015" in names[0]
            alph = 0.015
        print(f"Running {key}, mean = {np.mean(mean[-100:])} +- {np.mean(std[-100:])}")
        print('--')
        x = np.arange(len(mean))
        clean = clean_label(key)
        plt.plot(x, mean, label=f"{clean}. "+r"$\alpha="+str(alph)+"$")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.title(f"Basis for 400 episodes, averaged over {len(alls)} runs")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    dir = '/'.join((d.split("/"))[:-1])
    mysavefig(os.path.join(dir, 'results_main.png'), dpi=400)
    plt.close()

if __name__ == '__main__':
    analyse_main('Acrobot')
    analyse_main('MC')