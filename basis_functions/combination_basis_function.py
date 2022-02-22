from typing import List
from basis_functions.basis_function import BasisFunction
from common.types import State

class CombinationBasisFunction(BasisFunction):
    def __init__(self, bfs: List[BasisFunction]) -> None:
        self.bfs = bfs
    
    def get_value(self, state: State) -> float:
        ans = 1
        for b in self.bfs:
            ans *= b.get_value(state)
        return ans