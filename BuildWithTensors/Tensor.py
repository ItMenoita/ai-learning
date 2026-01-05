import numpy as np
from typing import List, Any
from numpy.typing import NDArray
from numbers import Number
from TensorMetadata import TensorMetadata
from TensorMetadata import Operation


def get_list_shape(data: Any) -> List[int]:
    """Calculates the shape of a rectangular nested list."""
    shape = []
    while isinstance(data, list):
        shape.append(len(data))
        if len(data) == 0:
            break
        data = data[0]
    return shape


class Tensor:
    def __init__(self, data: Any, requireGrad=False):
        if isinstance(data, np.ndarray):
            self.data: NDArray[np.number] = data
            self.shape: List[int] = list(data.shape)  # [2,3] a matrix 2 row by 3
        else:
            self.data: NDArray[np.number] = np.array(data)
            self.shape: List[int] = get_list_shape(data)

        self.requireGrad = requireGrad

        # Initialize grad with as none to not waste memory
        self.grad: NDArray[np.number] = None
        self._ctx: TensorMetadata = None

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            if other.shape != self.shape:
                raise ValueError("Can't add different shaped tensors")
            output = Tensor(
                self.data + other.data,
                requireGrad=True if self.requireGrad or other.requireGrad else False,
            )
            output._ctx = TensorMetadata([self, other], operation=Operation.ADD)
            return output
        else:
            raise TypeError("Can't add other object other then a tensor")

