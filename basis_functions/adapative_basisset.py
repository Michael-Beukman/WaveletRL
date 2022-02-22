import numpy as np
from common.types import State
from adaptive.awr.awr_candidate_groups import AWRCandidateGroups
from typing import List
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.adapative_basisset_simple import AdaptiveBasisSetSimple
from basis_functions.basis_set import BasisSet


class AdaptiveBasisSet(AdaptiveBasisSetSimple):
    def __init__(self, num_observations: int, awr_split_tolerance: float):
        super().__init__(num_observations, awr_split_tolerance)
        if awr_split_tolerance != -1:
            self.candidate_groups = AWRCandidateGroups(self, awr_split_tolerance)