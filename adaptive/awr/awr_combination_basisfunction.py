from typing import List

import numpy as np
from common.types import State
from basis_functions.wavelet.bspline.bspline_basisfunction import BSplineBasisFunction
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from common.utils import intersect_intervals

class AWRCombinationBasisFunction(AWRBasisFunction):
    """A basis function that contains the combination of 2 basis functions, each of which is a single AWRBasisFunction
    
    For example, self.bfs = [
        BSpline(dim=0, trans=0, scale=1),
        BSpline(dim=1, trans=1, scale=1)
    ]

    See `AWRBasisFunction` for documentation regarding the functions
    """
    def __init__(self, bfs: List[AWRBasisFunction], force_flat: bool = False):
        """
        Args:
            bfs (List[AWRBasisFunction]): The basis functions to use
            force_flat (bool, optional): If this is true, then, if bfs contains a `AWRCombinationBasisFunction`, we flatten it to make self.bfs *only* contain single basis functions. Defaults to False.
        """
        self.bfs = bfs
        if force_flat:
            self.bfs = AWRCombinationBasisFunction._unflatten_bfs(self)
            self.bfs = sorted(self.bfs, key=lambda x: x.dimension)
            
        super().__init__()

    @staticmethod
    def _unflatten_bfs(func: AWRBasisFunction) -> List[AWRBasisFunction]:
        # Recursively flattens the basis function set.
        all = []
        if isinstance(func, AWRCombinationBasisFunction):
            for f in func.bfs:
                all += AWRCombinationBasisFunction._unflatten_bfs(f)
        else:
            all.append(func)
        return all
                

    def get_omega(self) -> float:
        ans = 1
        for bf in self.bfs:
            ans *= bf.get_omega()
        return ans
    
    def is_supported(self, state: State) -> bool:
        ans  = 1
        for bf in self.bfs:
            ans = ans and bf.is_supported(state)
        return ans
    
    def get_children_funcs(self, dim: int) -> List['AWRBasisFunction']:
        children = self.bfs[dim].get_children_funcs(None)
        li = []
        for i in range(len(children)):
            li.append(
                AWRCombinationBasisFunction(
                    self.bfs[:dim] + [children[i]] + self.bfs[dim + 1:] 
                )
            )
        return li

    def get_children_weights(self, dim: int) -> List[float]:
        return self.bfs[dim].get_children_weights(None)
    
    def __repr__(self) -> str:
        return f"AWRCombinationBasisFunction(bs={self.bfs})"

    def get_value(self, state: State) -> float:
        ans = 1
        for bf in self.bfs:
            ans *= bf.get_value(state)
        return ans
    
    
    def get_dimensions(self, ndims: int) -> List[bool]:
        ans = np.zeros(ndims, dtype=np.bool8)
        for bf in self.bfs:
            ans |= bf.get_dimensions(ndims)
        return ans
    
    def interval_nonzero(self, ndims: int) -> List[List[float]]:
        ans = [[] for _ in range(ndims)]
        for bf in self.bfs:
            a = bf.interval_nonzero(ndims)
            for i in range(ndims):
                if len(ans[i]) == 0: ans[i] = a[i]
                elif len(a[i]) != 0: ans[i] = intersect_intervals(ans[i], a[i])
        return ans