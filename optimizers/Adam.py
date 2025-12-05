from core.Scalar import Scalar
import math


class Adam:

    def __init__(
        self,
        params: list[Scalar],
        momentumBeta=0.9,
        velocityBeta=0.999,
        learningRate=0.001,
        epsilon=10**-8,
    ):
        self.state = {}
        self.t = 0
        self.momentumBeta = momentumBeta
        self.velocityBeta = velocityBeta
        self.learningRate = learningRate
        self.epsilon = epsilon

        for p in params:
            self.state[p] = (0, 0)  # momentum , velocity

    def step(self) -> int:
        self.t += 1

        p: Scalar
        for p, (previousMomentum, previousVelocity) in self.state.items():
            momentum = (
                self.momentumBeta * previousMomentum + (1 - self.momentumBeta) * p.grad
            )
            velocity = self.velocityBeta * previousVelocity + (
                1 - self.velocityBeta
            ) * (p.grad**2)

            correctedMomentum = momentum / (1 - self.momentumBeta**self.t)
            correctedVelocity = velocity / (1 - self.velocityBeta**self.t)

            self.state[p] = (momentum, velocity)

            p.value = p.value - self.learningRate * (
                correctedMomentum / (math.sqrt(correctedVelocity) + self.epsilon)
            )

        return self.t

    def zeroGrad(self):
        p: Scalar
        for p, _ in self.state.items():
            p.grad = 0