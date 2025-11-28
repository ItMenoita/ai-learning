from Neuron import Neuron
from Scalar import Scalar
from typing import Union
from numbers import Number

class Layer:

    def __init__(self, inputSize, layerSize):
        self.neurons: list[Neuron] = []
        self.inputSize = inputSize

        for _ in range(layerSize):
            self.neurons.append(Neuron(inputSize))

    def __call__(self, inputs: list[Union[Scalar, Number]]) -> list[Scalar]:
        if len(inputs) != self.inputSize:
            raise ValueError(
                "Input size is different from expected: "
                + str(len(inputs))
                + " != "
                + str(self.inputSize)
            )

        result: list[Scalar] = []
        for neuron in self.neurons:
            result.append(neuron(inputs))

        return result

    def parameters(self) -> list[Scalar]:
        parameters = []
        for neuron in self.neurons:
            parameters+= neuron.parameters()
        return parameters