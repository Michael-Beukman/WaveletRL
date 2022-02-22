import numpy as np
from common.types import State
from typing import List
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.basis_set import BasisSet


class AdaptiveBasisSetSimple(BasisSet):
    def __init__(self, num_observations: int, awr_split_tolerance: float):
        self.dimensions = num_observations
        self.awr_split_tolerance = awr_split_tolerance
        self.functions: List[AWRBasisFunction] = self.get_functions()
        super().__init__(num_observations)

    
    def get_children_functions(self) -> List[List[List[AWRBasisFunction]]]:
        raise NotImplementedError 

    def get_functions(self) -> List[AWRBasisFunction]:
        raise NotImplementedError

    
    def get_value(self, state: State) -> np.ndarray:
        return np.array([f.get_value(state) for f in self.functions])
    
    def num_features(self) -> int:
        return len(self.functions)