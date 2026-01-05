from core.MLP import MLP
from core.Scalar import Scalar
from functions.LossFunctions import LossFunctions
from optimizers.Adam import Adam
from monitoring.LinearVisualizer import LinearVisualizer

##Dataset

# Inputs (3 numbers per example)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

# Desired Targets (1 number per example)
ys = [1.0, -1.0, -1.0, 1.0]

##Model
model = MLP(3, [4, 4, 1])

##Trainning

learningGradient = 0.01
epochs = 10000
optimizer = Adam(model.parameters())
visualizer = LinearVisualizer()

for i in range(epochs):
    result = []

    for data in xs:
        result.append(model(data))

    loss: Scalar = LossFunctions.meanSquaredError(result, ys)

    print(f"Epoch {str(i)} Loss: {str(loss)}")

    loss.backward()
    visualizer.addValue("Loss", loss.value)
    
    optimizer.step()
    optimizer.zeroGrad()

visualizer.plot()

# print the parameters after training
for p in model.parameters():
    print(f"Parameter value: {p.value}")
