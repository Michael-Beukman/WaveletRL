from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
import math
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBSplineBasisFunction
from sre_parse import State

import numpy as np
from basis_functions.adapative_basisset import AdaptiveBasisSet
from adaptive.awr.awr_candidate_groups import AWRCandidateGroups
from adaptive.ibfdd.ibfdd_candidate_groups import IBFDD_CandidateGroups
from typing import List, Tuple
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.basis_set import BasisSet


class IBFDDBSplineBasisSet(AdaptiveBasisSet):
    """This is an adaptive basis set that uses IBFDD to adapt.
    """
    def __init__(self, order: int, scale: int, num_observations: int, awr_split_tolerance: float, ibfdd_add_tolerance: float):
        self.order = order
        self.scale = scale
        awr_split_tolerance = -1
        super().__init__(num_observations, awr_split_tolerance)
        self.ibfdd_add_tolerance = ibfdd_add_tolerance
        self.ibfdd_candidates = IBFDD_CandidateGroups(self, self.ibfdd_add_tolerance)

    def get_children_functions(self) -> List[List[List[AWRBasisFunction]]]:
        # Does not really have children
        return []

    def discover(self, state: State, is_terminal: bool, delta: float, features: np.ndarray, weights: np.ndarray, eligibility_traces: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
            Performs the check for IBFDD given the state transition information.
            
            This returns (num_funcs_added, new_weights, new_elig_traces)
        """
        self.ibfdd_candidates.update_all(delta, state)
        should, indices = self.ibfdd_candidates.should_add()
        if not should:
            return 0, weights, eligibility_traces

        # ====================================

        new_func: AWRBasisFunction = self.ibfdd_candidates.candidate_functions[indices]
        new_weights = [0];
        if IBFDD_CandidateGroups.verbose:
            pass

        """
        need to remove
        1. Weights
        2. Eligibility traces
        3. self.functions
        """
        # 1, 2, 3

        # this call below takes care of adding the new conjunctions too
        self.ibfdd_candidates.remove_children(indices)

        """
        Need to add: 
        1. weights
        2. eligibility traces
        3. self.functions
        """

        weights = np.append(weights, 0)
        eligibility_traces = np.append(eligibility_traces, 0)
        self.functions.append(new_func)


        self._shrink = np.ones(self.num_features())
        return len(new_weights), weights, eligibility_traces
    
    def get_basic_phi_wavelets(self, scale):
        num_locs = self.order + 2 ** scale
        wavelets = [
            [None for _ in range(self.dimensions)]
            for __ in range(num_locs)
        ]
        for dim in range(self.dimensions):
            for loc in range(-self.order, 2 ** scale):
                wavelets[loc + self.order][dim] = AWRBSplineBasisFunction(
                    scale, loc, dim, self.order)
        return wavelets

    def get_functions(self) -> List[AWRBasisFunction]:
        ans: List[AWRBasisFunction] = []
        basic_phi_functions = self.get_basic_phi_wavelets(self.scale)
        for i in  range(len(basic_phi_functions)):
            for j in range(self.dimensions):
                ans.append(basic_phi_functions[i][j])
        return ans

    def __repr__(self) -> str:
        return f"IBFDDBSplineBasisSet(num_observations={self.num_observations}, order={self.order}, scale={self.scale}, awr_split_tolerance={self.awr_split_tolerance}, ibfdd_add_tolerance={self.ibfdd_add_tolerance})"
    
    