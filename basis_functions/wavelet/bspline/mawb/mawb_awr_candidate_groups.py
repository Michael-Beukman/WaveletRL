from basis_functions.adapative_basisset_simple import AdaptiveBasisSetSimple as AdaptiveBasisSet
from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
from typing import List, Tuple, Union

from common.types import State
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.basis_set import BasisSet
from basis_functions.combination_basis_function import CombinationBasisFunction
from adaptive.relevance_data import RelevanceData
import numpy as np


class MAWB_AWRCandidateGroups:
    verbose = False
    """
    This is basically a carbon copy of ../awr/AWRCandidateGroups for use with MAWB. This is just to allow for changes to be decoupled.
    """

    def __init__(self, parent: AdaptiveBasisSet, split_tolerance=0.1):
        self.split_tolerance = split_tolerance
        self.parent: AdaptiveBasisSet = parent

        # make the normal functions

        # make the eligible children

        # main functions' relevances and stuff
        self.main_relevance: RelevanceData = RelevanceData(self.parent.functions)

        # make the candidate functions
        # now relevances for candidates & do candidates themselves

        # candidates: shape of [actions, functions, dims, order+2]
        """
        candidates: shape of [functions, dims, order+2]
        """
        self.candidate_functions: List[List[List[AWRBasisFunction]]
                                       ] = self.parent.get_children_functions()

        # now the relevances for the candidates
        self.candidate_relevances: List[List[RelevanceData]] = \
            [
            [
                RelevanceData(self.candidate_functions[f_index][d]) for d in range(len(self.candidate_functions[f_index]))
            ] for f_index, f in enumerate(self.parent.functions)
        ]

    def update_all(self, delta: float, s: State):
        # first normal
        self.main_relevance.update_all(delta, s)
        for i in range(len(self.candidate_relevances)):
            for k in range(len(self.candidate_relevances[i])):
                self.candidate_relevances[i][k].update_all(delta, s)

    def should_split(self) -> Tuple[bool, Union[None, Tuple[int, int]]]:
        """
        Returns: bool, tuple, where tuple is None if bool is false. otherwise tuple = (function_index, dimension) to perform the split in
        The value is the maximum value of observed error - relevance
        """
        difference = []
        max_value = -1
        best_index = 0

        tmp = []
        observeds = self.main_relevance.observed_error
        rels = self.main_relevance.relevance
        # Find which function has the biggest difference between the observed error and relevance.
        for i in range(len(self.main_relevance.observed_error)):
            val = observeds[i] - (rels[i])
            tmp.append(val)
            if val > max_value:
                best_index = i
                max_value = val
        difference.append(tmp)

        best_function = best_index
        if max_value > self.split_tolerance:
            # We need to now replace functions[max_index] with it's children in some dimension
            # We need to find the largest value of the children relevances per dimension.
            maximum_average = -100
            max_d = -1
            for d in range(len(self.candidate_relevances[best_function])):
                val = np.mean((self.candidate_relevances[best_function][d].relevance))
                if val > maximum_average:
                    maximum_average = val
                    max_d = d
            return True, (best_function, max_d)
        return False, None

    def remove_main(self, index: int):
        self.main_relevance.remove_one(index)

    def remove_children(self, index: int, dim: int):
        """
        Removes the candidate relevances and the children functions
        :param action:
        :param index:
        :param dim:
        :return:
        """
        # we actually need to remove this whole one
        self.candidate_relevances.pop(index)
        self.candidate_functions.pop(index)

        assert len(self.candidate_relevances) == len(self.candidate_functions)

    def add_function(self, index: int, new_funcs: List[List[AWRBasisFunction]]):
        assert len(self.candidate_relevances) == index

        def get_scale(f):
            if isinstance(f, AWRCombinationBasisFunction):
                return max([get_scale(s) for s in f.bfs])
            else:
                return f.scale

        to_add = []
        for tmp in new_funcs:
            for f in tmp:
                if get_scale(f) > 4:
                    print("Scale > 4")
                    break
            else:
                to_add.append(tmp)

        new_funcs = to_add

        new_rel = [RelevanceData(new_funcs[d]) for d in range(len(new_funcs))]
        self.candidate_relevances.append(new_rel)
        self.candidate_functions.append(new_funcs)

        self.main_relevance.add_one(None)
        assert len(self.candidate_relevances) == len(self.candidate_functions)
        assert len(self.candidate_relevances) == len(self.main_relevance.relevance)

    def get_flat(self, action) -> List[AWRBasisFunction]:
        ans = []
        for j in self.candidate_functions[action]:
            for k in j:
                for l in k:
                    ans.append(l)
        return ans
