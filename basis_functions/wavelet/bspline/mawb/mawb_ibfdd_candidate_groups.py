import gym
from basis_functions.adapative_basisset_simple import AdaptiveBasisSetSimple as AdaptiveBasisSet
from adaptive.awr.awr_combination_basisfunction import AWRCombinationBasisFunction
from typing import List, Tuple, Union
from basis_functions.wavelet.bspline.awr.awr_bspline_basisfunction import AWRBSplineBasisFunction
from common.types import State
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.basis_set import BasisSet
from basis_functions.combination_basis_function import CombinationBasisFunction
from adaptive.relevance_data import RelevanceData
import numpy as np
from common.utils import intersect_intervals


def can_add(a: AWRBasisFunction, b: AWRBasisFunction, ndims):
    adim = a.get_dimensions(ndims)
    bdim = b.get_dimensions(ndims)
    
    asup = a.interval_nonzero(ndims)
    bsup = b.interval_nonzero(ndims)
    for i in range(len(adim)):
        if adim[i] and bdim[i]: return False
        if len(asup[i]) == 0 or len(bsup[i]) == 0: continue
        
        if len(intersect_intervals(asup[i], bsup[i])) == 0:
            return False
    return True
    


class MAWB_IBFDD_CandidateGroups:
    verbose = False
    """
    This is basically a carbon copy of ../ibfdd/IBFDD_CandidateGroups for use with MAWB. This is just to allow for changes to be decoupled.
    """

    def __init__(self, parent: AdaptiveBasisSet, tolerance=0.1, do_comb=False):
        """
        IF do comb is true, use a combinationBasisFunction as the basis, otherwise use SmartSmallComb...
        :param parent:
        :param tolerance:
        :param do_comb:
        """
        assert do_comb == False
        self.do_comb = do_comb

        self.tolerance = tolerance
        self.parent: AdaptiveBasisSet = parent
        # IFBDD does not need main relevances, as it is only concerned about the candidates.

        # make the candidate functions
        # now relevances for candidates & do candidates themselves

        # candidates: shape of [actions, functions, dims, order+2]
        """
        candidates: shape of [actions, functions, dims, order+2]
        """

        # one for each action
        self.candidate_functions: List[AWRBasisFunction] = self._get_conjunctions()

        # now the relevances for the candidates
        self.candidate_relevances: RelevanceData = RelevanceData(self.candidate_functions)

        self.max_relevance = None


        # This is a list of functions that were added to the basis set using IBFDD, *including* the initial decoupled basis set.
        self.conjunctions_added = [
            f for f in self.parent.functions
        ]

    def _get_conjunctions(self) -> List[AWRBasisFunction]:
        """
        This returns the conjunctions of all the function in the parent set.
        2d list is [action, function]
        :return:
        """
        tmp = []
        for i, f1 in enumerate(self.parent.functions):
            for j, f2 in enumerate(self.parent.functions[i+1:]):
                if not can_add(f1, f2, self.parent.dimensions):
                        # print("Cannot add in conjunctions", f1, f2)
                        continue

                if self.do_comb:
                    new_func = CombinationBasisFunction(f1, f2)
                else:
                    new_func = AWRCombinationBasisFunction([f1, f2], force_flat=True)
                tmp.append(new_func)
        return tmp

    def update_all(self, delta: float, s: State):
        """
        Updates the relevance of the specific action's functions
        :param delta:
        :param s:
        :param a:
        :return:
        """
        self.candidate_relevances.update_all(delta, s)

    def should_add(self):
        """
        Returns whether or not we should add a function to the set
        Returns: bool, tuple, where tuple is None if bool is false. otherwise tuple =
        (action, function_index) to add to the main set.
        :return:
        """
        difference = []
        max_value = -1
        best_index = (0, 0)
        tmp = []
        # for i in range(len(self.parent.functions[a])):
        for i in range(len(self.candidate_relevances.relevance)):
            # IBFDD uses just the relevance, not O - r
            val = self.candidate_relevances.relevance[i]
            # adding this abs makes it very slow and there is no abs in the thesis.
            val = abs(val)
            # tmp.append(val)
            if val > max_value:
                best_index = (i)
                max_value = val
        difference = (tmp)

        best_function = best_index
        if False:
            print("MAX INDEX", best_index, best_action, best_function, max_value)

        if max_value > self.tolerance:
            # We need to just add this function to the set. No dimension worries.
            self.max_relevance = max_value
            return True, (best_function)
        return False, None

    def _check_inside(self, func: AWRBasisFunction, list_of_funcs: List[AWRBasisFunction]) -> bool:
        # Returns True if this function is in the list of funcs
        str_func = str(func)
        for f in list_of_funcs:
            if str_func == str(f): return True
        return False
    
    def remove_children(self, index: int):
        """
        Removes the candidate relevances and the children functions.
        It also adds the pairwise conjunctions of this function and all the existing functions
        :param action:
        :param index:
        :return:
        """
        function_to_be_added = self.candidate_functions[index]
        all_conjunctions = []
        for f in self.conjunctions_added:
            if not can_add(f, function_to_be_added, self.parent.dimensions):
                # print("Cannot add")
                continue
            cand_conj = AWRCombinationBasisFunction([f, function_to_be_added], force_flat=True)
            
            # Now, just check if this conjunction is in:
            #   list of conjunctions_added
            #   list of self.parent.functions
            if self._check_inside(cand_conj, self.conjunctions_added):
                # print("Cannot add because already in conjunctions added")
                continue
            if self._check_inside(cand_conj, self.parent.functions):
                # print("Cannot add because already in parent.functions")
                continue

            all_conjunctions.append(cand_conj)

        
        self.conjunctions_added.append(function_to_be_added)
        # get the new conjunctions

        # we actually need to remove the relevance associated with it!
        self.candidate_functions.pop(index)
        self.candidate_relevances.remove_one(index)
        # remove the candidate function


        # Now we need to add the new pairwise conjunctions
        for f in all_conjunctions:
            self.candidate_relevances.add_one(f)

    def get_flat(self, action)-> List[AWRBasisFunction]:
        return self.candidate_functions[action]
