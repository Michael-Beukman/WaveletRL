from common.types import State


class BasisFunction:
    """This is a single basis function, i.e. one wavelet, or one combination, etc.
    """
    def __init__(self):
        pass


    def get_value(self, state: State) -> float:
        raise NotImplementedError()


    def __repr__(self) -> str:
        raise NotImplementedError()