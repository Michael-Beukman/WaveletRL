from basis_functions.combination_basis_function import CombinationBasisFunction
import math
from basis_functions.wavelet.bspline.bspline_basisfunction import BSplineBasisFunction
import numpy as np
from common.types import State
from basis_functions.basis_set import BasisSet


class FixedBSplineBasisSet(BasisSet):
    def __init__(self, num_observations: int, order: int, scale: int):
        """A Bspline Basis set. This has (self.order + 2**self.scale) ** self.num_observations number of functions.
                The order is similar to the degree of a polynomial. For example, order 0 is basically constant, order 3 is cubic, etc.
                The scale is by how much is the function dilated, similar to the a term in cos(a pi), except that this function is not periodic.

                The idea of this basis set is to create the following:
                [
                    b(scale, order, translation=a1, dimension=0) * b(scale, order, translation=a1, dimension=1),
                    b(scale, order, translation=a2, dimension=0) * b(scale, order, translation=a2, dimension=1),
                ]

                etc, for multiple translations until we fully support the interval [0, 1]. 
                    Each dimension is multiplied with other ones to measure interdependence.

        Args:
            num_observations (int): The number of state feature variables. E.g. mountain car = 2.
            order (int): 
            scale (int): 
        """
        self.order = order
        self.scale = scale
        self.dimensions = num_observations
        super().__init__(num_observations)
        self.functions = self.get_functions()
        assert len(self.functions) == self.num_features(), f"{len(self.functions)} != {self.num_features()}"
    
    def get_value(self, state: State) -> np.ndarray:
        return np.array([
            f.get_value(state) for f in self.functions
        ])
    
    def num_features(self) -> int:
        """Number of features we have

        Returns:
            int: 
        """
        return (self.order + 2**self.scale) ** self.num_observations


    def get_basic_phi_wavelets(self, scale):
        num_locs = self.order + 2 ** scale
        wavelets = [
            [None for _ in range(self.dimensions)]
            for __ in range(num_locs)
        ]
        for dim in range(self.dimensions):
            for loc in range(-self.order, 2 ** scale):
                wavelets[loc + self.order][dim] = BSplineBasisFunction(scale, loc, dim, self.order)
        return wavelets


    def get_index_lattice(self, scale):
        numLocations = pow(2, scale) + self.order
        xSize = pow(numLocations, self.dimensions)
        lattice = np.zeros((xSize, self.dimensions))
        for i in range(xSize):
            icopy = i
            for j in range(self.dimensions - 1, -1, -1):
                assert 0 <= j < self.dimensions
                lattice[i][self.dimensions - j - 1] = math.floor(icopy / pow(numLocations, j))
                icopy = icopy % pow(numLocations, j)
        lattice = np.array(lattice, dtype=np.int32)
        return lattice

    
    def get_functions(self):
        terms = []
        lattice = self.get_index_lattice(self.scale)
        basic_phi_wavelets = self.get_basic_phi_wavelets(self.scale)
        for i in range(len(lattice)):
            b = basic_phi_wavelets[lattice[i][0]][0] # already have dimension 1
            li_of_funcs = [b]
            for j in range(1, self.dimensions):
                li_of_funcs.append(basic_phi_wavelets[lattice[i][j]][j])
            b = CombinationBasisFunction(li_of_funcs)
            terms.append(b)
        return terms        
    
    def __repr__(self) -> str:
        return f"FixedBSplineBasisSet(num_observations={self.num_observations}, order={self.order}, scale={self.scale})"