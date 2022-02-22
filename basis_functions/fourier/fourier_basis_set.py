import numpy as np
from common.types import State
from basis_functions.basis_set import BasisSet

class FourierBasisSet(BasisSet):
    """
        Fourier Basis Function
    """
    def __init__(self, num_observations: int, order: int):
        self.order = order
        super().__init__(num_observations)
        
        dims = num_observations
        self.coeffs = np.indices((order+1,) * dims).reshape((dims, -1)).T
        coeff_norms = np.linalg.norm(self.coeffs, ord=2, axis=1)
        coeff_norms[0] = 1.0
        self._shrink = 1.0 / coeff_norms
    
    def get_value(self, state: State) -> np.ndarray:
        return np.cos(np.pi * np.dot(self.coeffs, state))

    def num_features(self) -> int:
        return (self.order+1) ** self.num_observations

    def __repr__(self) -> str:
        return f"FourierBasisSet(num_observations={self.num_observations}, order={self.order})"