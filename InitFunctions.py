from Scalar import Scalar
import random


class InitFunctions:

    @staticmethod
    def uniformWeights(size: int) -> list[Scalar]:
        if size < 0:
            raise ValueError("Size must be bigger than 0: " + str(size))
        weights = []
        for i in range(size):
            weights.append(Scalar(random.uniform(-1, 1)))
        return weights
