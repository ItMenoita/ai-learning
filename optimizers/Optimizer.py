from core.Scalar import Scalar

class Optimizer:

    def zeroGrad(value : Scalar):
        value.grad = 0
