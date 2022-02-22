from typing import List
from common.types import State
from basis_functions.basis_function import BasisFunction


class AWRBasisFunction(BasisFunction):
    """This is a basis function that can support the AWR algorithm

    """

    def get_omega(self) -> float:
        """Returns || Omega ||, i.e. the size of the interval where we are non zero.

        Returns:
            float: 
        """
        raise NotImplementedError()

    def is_supported(self, state: State) -> bool:
        """Is my value for this state non zero? 

        Args:
            state (State): The state

        Returns:
            bool
        """
        raise NotImplementedError()
    
    def get_children_funcs(self, dim: int) -> List['AWRBasisFunction']:
        """
            Returns a list of children functions.
        """
        raise NotImplementedError()

    def get_children_weights(self, dim: int) -> List[float]:
        """Returns a list with the same length as get_children_funcs, that contain their weights.
            These weights sum to 1.
        Args:
            dim (int): 

        Returns:
            List[float]: 
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        raise NotImplementedError()

    def get_dimensions(self, ndims: int) -> List[bool]:
        """Returns a one hot encoded vector for which dimensions this is active in
            ndims: How many dims are there in total
        Returns:
            List[bool] 1 if active in that dimension, 0 otherwise
        """
        return []
    
    def interval_nonzero(self, ndims: int) -> List[List[float]]:
        """Returns a list, each element is an interval representing the nonzero interval for that dimension.

        Returns:
            List[List[float]]: 
        """
        return [];