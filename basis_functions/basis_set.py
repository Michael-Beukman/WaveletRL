from typing import Tuple, final
from common.types import State
import numpy as np


class BasisSet:
    """
        A basic implementation of a basis set. This specifically takes in states and transforms them into 
        more complex representations. When having multiple actions, then you should have multiple instances of this basis set, 
        one for each action.
    """
    def __init__(self, num_observations: int):
        self.num_observations = num_observations
        self._shrink = np.ones(self.num_features())
    
    def get_value(self, state: State) -> np.ndarray:
        """This transforms the state into an array of length self.num_features() representing the value of the basis functions at this state.

        Args:
            state (State): The state. It must be of shape (num_observations, ) and must also be normalised to between 0 and 1.

        Returns:
            np.ndarray: Transformed input.
        """
        raise NotImplementedError()
    
    @final
    def get_shrink(self) -> np.ndarray:
        """This returns the 'shrink', or correction term used when updating weights.
             For example, for the Fourier basis, this correction term is 1/coefficient_norms.

        Returns:
            np.ndarray: The shrink. This should be of length self.num_features() and is all ones by default.
        """
        return self._shrink
    
    def discover(self, state: State, is_terminal: bool, delta: float, features: np.ndarray, 
                    weights: np.ndarray, eligibility_traces: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """This allows the basis function method to be adaptive, in that it can add basis functions, as in ATC, AWR, MAWB, IBFDD, etc.
             These methods can change the weights and eligibility traces, so these must be passed in and returned.

        Args:
            state (State): The current state
            is_terminal (bool): Is the current state terminal.
            delta (float): The TD error from this timestep.
            features (np.ndarray): The transformed features gotten by calling self.get_value(state). This is to save recomputing them again.
            weights (np.ndarray): Array of shape (self.num_features()) representing the weights as they are at the moment.
            eligibility_traces (np.ndarray): Array of shape (self.num_features()) representing the eligibility traces as they are at the moment.

        Returns:
            Tuple[int, np.ndarray, np.ndarray]: (number of functions added, new weights, new eligibility traces.)

            Any changes to weights (for example if we have new functions, they need weights), and eligibility traces must be made in this function
            and returned.
        """
        return 0, weights, eligibility_traces

    def num_features(self) -> int:
        """How many features do we have, i.e. what is the length of self.get_value()?

        Returns:
            int: Number of features.
        """
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        raise NotImplementedError()