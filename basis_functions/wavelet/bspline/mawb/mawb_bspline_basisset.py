from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
import math
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBSplineBasisFunction
from sre_parse import State

import numpy as np
from basis_functions.adapative_basisset_simple import AdaptiveBasisSetSimple

from basis_functions.wavelet.bspline.mawb.mawb_awr_candidate_groups import MAWB_AWRCandidateGroups
from basis_functions.wavelet.bspline.mawb.mawb_ibfdd_candidate_groups import MAWB_IBFDD_CandidateGroups
from typing import List, Tuple
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.basis_set import BasisSet


class MAWBSplineBasisSet(AdaptiveBasisSetSimple):
    """
        This is a basisset that combines AWR & IBFDD to get MAWB. The code is largely copied from those classes.
    """
    def __init__(self, order: int, scale: int, num_observations: int, awr_split_tolerance: float, ibfdd_add_tolerance: float):
        self.order = order
        self.scale = scale
        self.ibfdd_add_tolerance = ibfdd_add_tolerance
        super().__init__(num_observations, awr_split_tolerance)
        self.candidate_groups = MAWB_AWRCandidateGroups(self, awr_split_tolerance)
        self.ibfdd_candidates = MAWB_IBFDD_CandidateGroups(self, self.ibfdd_add_tolerance)
        self.ibfdd_added = 0
        self.awr_added = 0

    def get_children_functions(self) -> List[List[List[AWRBasisFunction]]]:
        return [
            [f.get_children_funcs(None)] for f in self.functions
        ]
    def discover(self, state: State, is_terminal: bool, delta: float, features: np.ndarray, weights: np.ndarray, eligibility_traces: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        # Do both 
        a, weights, eligibility_traces = self.discover_ibfdd(state, is_terminal, delta, features, weights, eligibility_traces)
        b, weights, eligibility_traces = self.discover_awr(state, is_terminal, delta, features, weights, eligibility_traces)
        
        self.ibfdd_added += a
        self.awr_added += b
        
        if a + b != 0:
            print(f"{a} IBFDD funcs | {b} AWR Funcs")
        return a + b, weights, eligibility_traces
        
    def discover_awr(self, state: State, is_terminal: bool, delta: float, features: np.ndarray, weights: np.ndarray, eligibility_traces: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        self.candidate_groups.update_all(delta, state)
        should, indices = self.candidate_groups.should_split()
        if not should:
            return 0, weights, eligibility_traces

        # otherwise, split now
        function_index, dimension = indices

        function_that_will_be_splitted = self.functions[function_index]
        if isinstance(function_that_will_be_splitted, AWRBSplineBasisFunction): dimension = None
        new_weights = function_that_will_be_splitted.get_children_weights(dimension)
        new_funcs = function_that_will_be_splitted.get_children_funcs(dimension)
        parent_weight = weights[function_index]

        # First remove the following for i = function_index
        """
        1. Weights[i]
        2. Eligibility traces[i]
        3. self.functions[i]
        4. self.candidate_groups.children[i]
        5. self.candidate_groups remove current relevance
        """

        weights = np.delete(weights, function_index)
        eligibility_traces = np.delete(eligibility_traces, function_index)

        self.functions.pop(function_index)

        self.candidate_groups.remove_main(function_index)
        self.candidate_groups.remove_children(function_index, dimension)

        """But, we need to add in the new functions too!
            1. New weights
            2. New eligibility traces
            3. new functions
            4. new candidate main
            5. new candidate children.
        """

        
        assert len(new_weights) == len(new_funcs)
        
        for w, f in zip(new_weights, new_funcs):
            weights, eligibility_traces = self._add_function(weights, eligibility_traces, f, w * parent_weight, 0)
        
        self._shrink = np.ones(self.num_features())
        return len(new_weights), weights, eligibility_traces

    def discover_ibfdd(self, state: State, is_terminal: bool, delta: float, features: np.ndarray, weights: np.ndarray, eligibility_traces: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        self.ibfdd_candidates.update_all(delta, state)
        should, indices = self.ibfdd_candidates.should_add()
        if not should:
            return 0, weights, eligibility_traces

        # ====================================

        new_func: AWRBasisFunction = self.ibfdd_candidates.candidate_functions[indices]
        new_weights = [0];

        """
        need to remove
        1. Weights
        2. Eligibility traces
        3. self.functions (use remove maybe)
        4. self.candidate_groups.children (remove one)
        4.1. self.candidate_groups.children (add new children) ONLY LATER
        5. self.candidate_groups remove current thingy relevances.
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
        
        weights, eligibility_traces = self._add_function(weights, eligibility_traces, new_func, 0, 0)


        self._shrink = np.ones(self.num_features())
        return len(new_weights), weights, eligibility_traces
    
    
    def _add_function(self, weights, eligibility_traces,
                      new_func: AWRBSplineBasisFunction, weight: float, el:float = 0):

        weights = np.append(weights, weight)
        eligibility_traces = np.append(eligibility_traces, el)
        index = len(self.functions)
        self.functions.append(new_func)

        if isinstance(new_func, AWRBSplineBasisFunction):
            self.candidate_groups.add_function(index, [new_func.get_children_funcs(None)])
        else:
            self.candidate_groups.add_function(index, [new_func.get_children_funcs(dim) for dim in range(len(new_func.bfs))])
        
        return weights, eligibility_traces
    
    
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
        return f"MAWB(num_observations={self.num_observations}, order={self.order}, scale={self.scale}, awr_split_tolerance={self.awr_split_tolerance}, ibfdd_add_tolerance={self.ibfdd_add_tolerance})"
    
    