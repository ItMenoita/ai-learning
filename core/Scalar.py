from numbers import Number
from typing import Optional, Union
from collections import deque
from math import tanh


# Parents     Child
#
#  X
#
#  +      =     Z
#
#  Y
#
class Scalar:

    def __init__(self, value: Number):
        self.value: Number = value
        self.grad: Optional[Number] = None
        self.parents: list[tuple[Scalar, Number]] = []  # [(parent, local_grad), ...]

    def getScalar(arg: Union[Number, "Scalar"]) -> "Scalar":
        scalarArg: Scalar

        if isinstance(arg, Scalar):
            scalarArg = arg
        elif isinstance(arg, Number):
            scalarArg = Scalar(arg)
        else:
            raise TypeError("Must be instance of Scalar")
        return scalarArg

    # Derivative of Z = X + Y is dZ/dX = 1, dZ/dY = 1
    def __add__(self, arg):
        sArg = Scalar.getScalar(arg)
        result = Scalar(self.value + sArg.value)
        result.parents = [[self, 1], [sArg, 1]]
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, arg):
        return self.__add__(arg * -1)

    def __rsub__(self, other):
        return self.__sub__(other)

    # Derivative of Z = X * Y is dZ/dX = Y, dZ/dY = X
    def __mul__(self, arg):
        sArg = Scalar.getScalar(arg)
        result = Scalar(self.value * sArg.value)
        result.parents = [[self, sArg.value], [sArg, self.value]]
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    # Derivative of Z = tanh(X) is dZ/dX = 1 - tanh(X)^2
    def tanh(self):
        tanhValue = tanh(self.value)
        result = Scalar(tanhValue)
        result.parents = [[self, 1 - (tanhValue * tanhValue)]]
        return result

    # Chain rule application
    # Parent Grad += My Grad * Local Derivative
    def backward(self):
        self.grad = self.grad or 1

        topology: deque = self.buildTopology()
        topology

        for currentNode in reversed(topology):
            parent: Scalar
            localgrad: Number

            for parent, localgrad in currentNode.parents:
                #                  the child influence on the output * how much parent influences the child
                parent.grad = (parent.grad or 0) + (currentNode.grad * localgrad)

    def buildTopology(self) -> deque:
        # Since python doesn't have a linked hash set........ we need two structures
        topology = deque()
        visited = set()

        def visit(node: Scalar):
            parent: Scalar
            for parent, _ in node.parents:
                if not parent in visited:
                    visit(parent)

            topology.append(node)
            visited.add(node)

        visit(self)
        return topology

    def __str__(self):
        return f"Scalar(value={self.value}, grad={self.grad})"


###
#             a
#       c  =  +
#             b
# d  =  +
#
#       b
#

# a = Scalar(2)
# b = Scalar(3)
# c = a + b
# d = c + b
# d.backward()
# print(f"a.grad: {a.grad}") # 1
# print(f"b.grad: {b.grad}") # 2
# print(f"c.grad: {c.grad}") # 1
# print(f"d.grad: {d.grad}") # 1


# ----

# e = a * b
# d = e + b
# d.backward()

# print(f"a.grad: {a.grad}")  # 3
# print(f"b.grad: {b.grad}")  # 3 (2 + 1)
# print(f"d.grad: {d.grad}")  # 1
# print(f"e.grad: {e.grad}")  # 1
