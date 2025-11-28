from MLP import MLP
from Scalar import Scalar
from LossFunctions import LossFunctions

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

for i in range(epochs):
    result = []

    for data in xs:
        result.append(model(data))

    loss: Scalar = LossFunctions.meanSquaredError(result, ys)

    print(f"Epoch {str(i)} Loss: {str(loss)}")

    loss.backward()

    p:Scalar
    
    for p in model.parameters():
        p.value -= p.grad*learningGradient
        p.grad = None

# print the parameters after training
for p in model.parameters():
    print(f"Parameter value: {p.value}")