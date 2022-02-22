import datetime
from typing import List

from matplotlib import pyplot as plt
import bz2
import pickle

def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def mysavefig(*args, **kwargs):
    args = list(args)
    kwargs['dpi'] = 400
    plt.savefig(*args, **kwargs)
    args[0] = args[0].split(".png")[0] + ".pdf"
    plt.savefig(*args, **kwargs)
    
    
def overlap(interval1, interval2):
    """
    E.g., Given [0, 4] and [1, 10] returns [1, 4].
    """
    if interval2[0] <= interval1[0] <= interval2[1]:
        start = interval1[0]
    elif interval1[0] <= interval2[0] <= interval1[1]:
        start = interval2[0]
    else:
        return [];

    if interval2[0] <= interval1[1] <= interval2[1]:
        end = interval1[1]
    elif interval1[0] <= interval2[1] <= interval1[1]:
        end = interval2[1]
    else:
        return []

    return [start, end]

def intersect_intervals(a: List[float], b: List[float]) -> List[float]:
    return overlap(a, b)

def clean_label(s):
    # Cleans the labels for plots.
    s = s.replace("AWRBsplineBasisSet", "AWR")
    s = s.replace("FixedBSplineBasisSet", "Fixed B-Spline")
    s = s.replace("IBFDDBSplineBasisSet", "IBFDD")
    s = s.replace(", awr_split_tolerance=-1", "")
    s = s.replace("FourierBasisSet", "Fourier")
    s = s.replace("num_observations=2, ", "")
    s = s.replace("num_observations=4, ", "")
    return s
  


# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data
