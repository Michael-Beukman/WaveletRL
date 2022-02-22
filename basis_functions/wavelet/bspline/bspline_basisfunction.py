from basis_functions.basis_function import BasisFunction
from common.types import State
from basis_functions.wavelet.bspline.bspline_phi import phi_normal
from math import sqrt
class BSplineBasisFunction(BasisFunction):
    """A BSpline Basis Function of a specific order
    """
    def __init__(self, scale: int, translation: int, dimension: int, order: int) -> None:
        self.scale = scale
        self.translation = translation
        self.order = order
        self.dimension = dimension
        self.multiplier = 2 ** scale
        
    def get_value(self, state: State) -> float:
        t = self.multiplier * state[self.dimension] - self.translation
        return phi_normal(t, self.order)