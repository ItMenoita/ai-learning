from typing import Union
from core.Layer import Layer
from core.Scalar import Scalar


class MLP:

    def __init__(self, inputSize: int, hiddenLayers: list[int]):
        self.layers: list[Layer] = []

        prevLayerSize = inputSize
        for layerSize in hiddenLayers:
            self.layers.append(Layer(prevLayerSize, layerSize))
            prevLayerSize = layerSize

    def __call__(self, inputs: list[Scalar]) -> Union[list[Scalar], Scalar]:
        inputToNextLayer = inputs
        for layer in self.layers:
            inputToNextLayer = layer(inputToNextLayer)

        return inputToNextLayer[0] if len(inputToNextLayer) == 1 else inputToNextLayer

    def parameters(self) -> list[Scalar]:
        parameters = []
        for layer in self.layers:
            parameters+= layer.parameters()
        return parameters
