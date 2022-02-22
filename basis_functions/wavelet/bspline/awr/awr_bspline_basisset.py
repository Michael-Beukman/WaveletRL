from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
import math
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBSplineBasisFunction
from sre_parse import State

import numpy as np
from basis_functions.adapative_basisset import AdaptiveBasisSet
from adaptive.awr.awr_candidate_groups import AWRCandidateGroups
from typing import List, Tuple
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.basis_set import BasisSet


class AWRBsplineBasisSet(AdaptiveBasisSet):
    """This is a basis set that performs AWR when needed.

    """
    def __init__(self, order: int, scale: int, num_observations: int, awr_split_tolerance: float):
        """
        Args:
            order (int): 
            scale (int): 
            num_observations (int): The number of actual state space dimensions, e.g. 2 for MountainCar
            awr_split_tolerance (float): Threshold for splitting
        """
        self.order = order
        self.scale = scale
        super().__init__(num_observations, awr_split_tolerance)

    def get_children_functions(self) -> List[List[List[AWRBasisFunction]]]:
        return [
            [f.get_children_funcs(d) for d in range(self.dimensions)] for f in self.functions
        ]

    def discover(self, state: State, is_terminal: bool, delta: float, features: np.ndarray, weights: np.ndarray, eligibility_traces: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
            Performs the check for AWR given the state transition information.
            
            This returns (num_funcs_added, new_weights, new_elig_traces)
        """
        self.candidate_groups.update_all(delta, state)
        should, indices = self.candidate_groups.should_split()
        if not should:
            return 0, weights, eligibility_traces

        # otherwise, split now
        function_index, dimension = indices

        function_that_will_be_splitted = self.functions[function_index]
        new_weights = function_that_will_be_splitted.get_children_weights(
            dimension)
        new_funcs = function_that_will_be_splitted.get_children_funcs(
            dimension)
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

        """But, we need to add in the new functions too,
            1. New weights
            2. New eligibility traces
            3. new functions
            4. new candidate main
            5. new candidate children.
        """

        # 1 and 2
        for w in new_weights:
            weights = np.append(weights, w * parent_weight)
            eligibility_traces = np.append(eligibility_traces, 0)

        # 3, 4 and 5.
        for f in new_funcs:
            index = len(self.functions)
            self.functions.append(f)
            self.candidate_groups.add_function(
                index, [f.get_children_funcs(dim) for dim in range(self.dimensions)])
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

    def get_index_lattice(self, scale):
        """This is used to populate all of the combinations of functions from different dimensions
        """
        numLocations = pow(2, scale) + self.order
        xSize = pow(numLocations, self.dimensions)
        lattice = np.zeros((xSize, self.dimensions))
        for i in range(xSize):
            icopy = i
            for j in range(self.dimensions - 1, -1, -1):
                assert 0 <= j < self.dimensions
                lattice[i][self.dimensions - j -
                           1] = math.floor(icopy / pow(numLocations, j))
                icopy = icopy % pow(numLocations, j)
        lattice = np.array(lattice, dtype=np.int32)
        return lattice

    def get_functions(self) -> List[AWRBasisFunction]:
        """
        Uses the above functions to create the full basis set.
        """
        terms = []
        lattice = self.get_index_lattice(self.scale)
        basic_phi_wavelets = self.get_basic_phi_wavelets(self.scale)
        for i in range(len(lattice)):
            b = basic_phi_wavelets[lattice[i][0]][0]
            li_of_funcs = [b]
            for j in range(1, self.dimensions):
                li_of_funcs.append(basic_phi_wavelets[lattice[i][j]][j])
            b = AWRCombinationBasisFunction(li_of_funcs)
            terms.append(b)
        return terms

    def __repr__(self) -> str:
        return f"AWRBsplineBasisSet(num_observations={self.num_observations}, order={self.order}, scale={self.scale}, awr_split_tolerance={self.awr_split_tolerance})"