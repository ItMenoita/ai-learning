from core.Scalar import Scalar
from numbers import Number


class LossFunctions:

    def meanSquaredError(prediction: list[Scalar], target: list[Number]) -> Scalar:
        if len(prediction) != len(target):
            raise ValueError(
                "Prediction size is different from target: "
                + str(len(prediction))
                + " != "
                + str(len(target))
            )

        loss = 0
        for i in range(len(prediction)):
            loss += (prediction[i] - target[i]) * (prediction[i] - target[i])

        return loss * (1 / len(prediction))
