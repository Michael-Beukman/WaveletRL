from typing import List

from common.types import State
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBasisFunction
import numpy as np

REL_MODE = 'PROPER'
class RelevanceData:
    """
    This class stores relevance data, i.e. relevance, sample_count and obs error for a list of functions.
    """
    def __init__(self, functions: List[AWRBasisFunction]):
        self.functions: List[AWRBasisFunction] = functions

        # Main values
        self._relevance: np.array = self.get_zeroes()
        self._sample_counts: np.array = self.get_zeroes().astype(int)
        self._observed_error: np.array = self.get_zeroes()
        
        self.previous_features:np.ndarray = np.array([])
        self.is_null = False;
        # Exponentially weighted moving average value.
        self.epsilon = 0.99
        self.omegas = []
        self.get_omegas()

    @property
    def relevance(self):
        if REL_MODE == "PROPER":
            return self._relevance
        elif REL_MODE == "ABS":
            return np.abs(self._relevance)
        ans = np.zeros_like(self._relevance)
        ans =  self.omegas * np.abs(self._relevance)
        return ans

    @property
    def sample_counts(self):
        return self._sample_counts
    
    @property
    def observed_error(self):
        if REL_MODE == "PROPER":
            return self._observed_error

        ans = np.zeros_like(self._observed_error)
        ans =  self.omegas * self._observed_error
        return ans

        return self._observed_error
    
    
    def get_zeroes(self) -> np.array:
        return np.zeros(len(self.functions), dtype=np.float64)

    def get_features(self, s: State):
        tmp_funcs = self.functions[:len(self._relevance)]
        features = np.zeros(len(tmp_funcs))
        for i, f in enumerate(tmp_funcs):
            if not f.is_supported(s): continue
            features[i] = f.get_value(s)
        return features

    def get_omegas(self, tmp_funcs=None):
        if len(self.omegas) != len(self._relevance):
            if tmp_funcs is None:
                tmp_funcs = self.functions[:len(self._relevance)]
            self.omegas = np.array([f.get_omega() for f in tmp_funcs])
        return self.omegas

    def update_all(self, delta:float, s:State):
        """
        Updates all the relevances. Uses the state to make sure to update it all.
        :return:
        """
        # first get features
        assert len(self.functions) == len(self._relevance), f"{len(self.functions)}, {len(self._relevance)}"
        tmp_funcs = self.functions[:len(self._relevance)]
        features = self.get_features(s)
        
        omegas = self.get_omegas(tmp_funcs)
        is_supported = np.array([f.is_supported(s) for f in tmp_funcs])

        samples = self._sample_counts
        relevance = self._relevance
        obs = self._observed_error
        
        if len(is_supported):
            tmp_val = self._get_updated_relevance(
                samples[is_supported], relevance[is_supported],
                omegas[is_supported], delta, features[is_supported]
            );
            # update only supported
            relevance[is_supported] = tmp_val

            self._observed_error = (self._get_updated_observation_error(
                samples, obs, omegas, delta, features
            ))
            self.previous_features = features
        self._sample_counts += is_supported != 0
        
    
    def _get_updated_relevance(self, sample_count:np.ndarray, relevance:np.ndarray, omegas:np.ndarray, delta: float, features:np.ndarray) -> np.ndarray:
        # return sample_count / (sample_count + 1) * relevance + 1 / (sample_count + 1) * omegas * delta * features
        # this is the EWMA (eq 4.4.)
        if REL_MODE == 'PROPER':
            return sample_count / (sample_count + 1) * (self.epsilon * relevance + 
                        omegas*(1-self.epsilon)  * delta * features)
        return self.epsilon * relevance + (1 - self.epsilon)  * delta * features

    def _get_updated_abs_error(self, sample_count: np.ndarray, abs_err: np.ndarray, omegas:np.ndarray, delta:float) -> np.ndarray:
        return (abs_err * (sample_count - 1) + omegas * abs(delta)) / np.clip(sample_count,1,None)  # don't div by 0. 

    def _get_updated_observation_error(self, sample_count: np.ndarray, observed_error: np.ndarray, omegas: np.ndarray, delta: float, features:np.ndarray) -> np.ndarray:
        #  Decayed, sample attenuated Observed error
        if REL_MODE == 'PROPER':
            return sample_count / (sample_count + 1) * (self.epsilon * observed_error + omegas*(1-self.epsilon)  * abs(delta) * features)
        return self.epsilon * observed_error + (1 - self.epsilon)  * np.abs(delta) * features        

    def __repr__(self):
        return f"{self._sample_counts, self._relevance, self._observed_error}"

    def remove_one(self, index):
        self._sample_counts = np.delete(self._sample_counts, index)  # pop(index)
        self._relevance = np.delete(self._relevance, index)
        self._observed_error = np.delete(self._observed_error, index)

        assert len(self._observed_error) == len(self.functions)
        self.get_omegas()
        pass

    def add_one(self, function: AWRBasisFunction):
        self._sample_counts = np.append(self._sample_counts, 0)
        self._relevance = np.append(self._relevance, 0)
        self._observed_error = np.append(self._observed_error, 0)
        if function is not None:
            self.functions.append(function)

        assert len(self.functions) == len(self._relevance)
        self.get_omegas()


    def makeNull(self):
        """
        Removes this

        :return:
        """
        self._sample_counts = np.array([])
        self._relevance = np.array([])
        self._observed_error = np.array([])
        self.functions = np.array([])
        self.previous_features = np.array([])
        self.is_null = True
