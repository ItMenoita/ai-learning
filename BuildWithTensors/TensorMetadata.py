from typing import List, Any
from enum import Enum


class Operation(Enum):
    ADD = 1


class TensorMetadata:

    def __init__(self, parents: List = [], operation: Operation = None):
        self.parent: List = parents  # List of tensors
        self.operation: Operation = operation
