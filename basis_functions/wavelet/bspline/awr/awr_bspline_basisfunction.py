import numpy as np
from common.types import State
from typing import List
from basis_functions.wavelet.bspline.bspline_basisfunction import BSplineBasisFunction
from adaptive.awr.awr_basisfunction import AWRBasisFunction
from basis_functions.wavelet.bspline.bspline_phi import phi_normal

class AWRBSplineBasisFunction(AWRBasisFunction):
    """A Bspline basis function that satisfies the AWRBasisFunction interface.

    Args:
        AWRBasisFunction (_type_): _description_
    """
    def __init__(self, scale: int, translation: int, dimension: int, order: int) -> None:
        self.scale = scale
        self.translation = translation
        self.order = order
        self.dimension = dimension
        self.multiplier = 2 ** scale

        a, b = self.translation / self.multiplier, ((self.order + 1 + self.translation) / self.multiplier)
        self.interval = [min(a, b), max(a, b)]

    def get_children_funcs(self, dim: int) -> List['AWRBasisFunction']:
        assert dim is None or dim == self.dimension, f"Bad dimension {dim} for this function ({self.dimension})"
        return [
            AWRBSplineBasisFunction(self.scale + 1, dimension=self.dimension, translation=t + 2 * self.translation, order=self.order) for t in
            range(self.order + 2)
        ]
    
    def get_children_weights(self, dim: int) -> List[float]:
        """When we split this function into children, how do we weight them? This returns one weight per child, s.t. the sum of the weights is one.
        """
        assert dim is None or dim == self.dimension, f"Bad dimension {dim} for this function ({self.dimension})"
        order = self.order
        weights = [0 for i in range(order + 2)];
        if (order == 0):
            weights[0] = 1;
            weights[1] = 1;
        elif (order == 1):
            weights[0] = 0.5;
            weights[1] = 1;
            weights[2] = 0.5;
        elif (order == 2):
            weights[0] = 0.25;
            weights[1] = 0.75;
            weights[2] = 0.75;
            weights[3] = 0.25;
        elif (order == 3):
            weights[0] = 0.125;
            weights[1] = 0.5;
            weights[2] = 0.75;
            weights[3] = 0.5;
            weights[4] = 0.125;
        elif (order == 4):
            weights[0] = 0.0625;
            weights[1] = 0.3125;
            weights[2] = 0.625;
            weights[3] = 0.625;
            weights[4] = 0.3125;
            weights[5] = 0.0625;

        return weights
    
    def get_omega(self) -> float:
        a, b = self.interval
        return min(b, 1) - max(a, 0)
    
    def get_value(self, state: State) -> float:
        t = self.multiplier * state[self.dimension] - self.translation
        return phi_normal(t, self.order)

    def is_supported(self, state: State) -> bool:
        s = state[self.dimension]
        return self.interval[0] <= s <= self.interval[1]
        

    def __repr__(self) -> str:
        return f"AWRBSplineBasisFunction(scale={self.scale}, order={self.order}, dimension={self.dimension}, translation={self.translation})"
    
    def get_dimensions(self, ndims: int) -> List[bool]:
        ans = np.zeros(ndims, dtype=np.bool8)
        ans[self.dimension] = True
        return ans
    
    def interval_nonzero(self, ndims: int) -> List[List[float]]:
        ans = [[] for _ in range(ndims)]
        ans[self.dimension] = self.interval
        return ans