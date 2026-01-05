from core.Scalar import Scalar
from typing import Callable
from functions.InitFunctions import InitFunctions
from functions.ActivationFunctions import ActivationFunction
from numbers import Number
from typing import Union


class Neuron:

    def __init__(
        self,
        inputSize: int,
        initFunc: Callable[int, list[Scalar]] = InitFunctions.uniformWeights,
        activationFunction: Callable[Scalar, Scalar] = ActivationFunction.tanh,
    ):
        self.weights: list[Scalar] = initFunc(inputSize)
        self.bias: Scalar = initFunc(1)[0]

        # If we define the activation function per neuron, we could have neurons within the same layer
        # using different activation functions. Would this actually be beneficial?

        # It would give us flexibility, which might be interesting, but it also introduces a drawback:
        # we lose the ability to efficiently vectorize operations on the GPU, since we would need to
        # loop through each neuron individually and call its activation function.

        # We could try to optimize this by grouping neurons that share the same activation function
        # and applying the activation in batches, but this would add complexity and may not provide
        # any meaningful performance improvement.

        # But for the sake of learning let's make the Neuron complete and the owner of this logic for now
        self.activationFunction = activationFunction

    def __call__(self, inputs: list[Union[Scalar, Number]]) -> Scalar:
        if len(inputs) != len(self.weights):
            raise ValueError(
                "Input size is different from expected: "
                + str(len(inputs))
                + " != "
                + str(len(self.weights))
            )

        result = self.bias
        for i in range(len(self.weights)):
            result += self.weights[i] * inputs[i]

        return self.activationFunction(result)

    def parameters(self) -> list[Scalar]:
        return self.weights + [self.bias]
