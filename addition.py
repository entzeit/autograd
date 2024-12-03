from autograd.neural_network import NeuralNetwork
from autograd.data_generator import DataGenerator
from autograd.tensor import Tensor
from autograd.loss import Loss
import random

seed = random.randint(0, 1000)
print(f"Seed: {seed}")
X_train, Y_train, X_test, Y_test = DataGenerator([(lambda a, b: a + b)], value_min=-100, value_max=100, binary=False, seed=seed).generate(normalize=True)

input_size = X_train[0].shape[0]
layers = [(1, Tensor.identity)]
loss_function = Loss.mse
lr = 0.1
epochs = 100
debug_steps = 10

model = NeuralNetwork(input_size, layers, lr, seed)
for epoch in range(epochs):
    Y_pred = model.forward(X_train, Y_train)
    loss = loss_function(Y_pred, Y_train)
    if ((epoch + 1) % (epochs/debug_steps) == 0) or epoch == 0:
            print(f"Epoch [{epoch + 1:>{len(str(abs(epochs)))}}/{epochs}]\tLoss: {loss.array[0][0]:.4f}")
    loss.backward()
    model.update()

model.test(X_test, Y_test, tolerance=0.1, binary=False)