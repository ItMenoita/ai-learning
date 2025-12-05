from numbers import Number
from typing import Dict, List

import matplotlib.pyplot as plt


class LinearVisualizer:

    def __init__(self):
        self.data: Dict[str, List[Number]] = {}

    def addValue(self, datasetName: str, value: Number):
        data = self.data.get(datasetName, [])
        data.append(value)
        self.data[datasetName] = data

    def plot(self, xAxisName = "Step", yAxisName ="Value", title = "Linear Visualizer"):
        if not self.data:
            return

        plt.figure()
        for name, values in self.data.items():
            plt.plot(range(len(values)), values, label=name)

        plt.xlabel(xAxisName)
        plt.ylabel(yAxisName)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
