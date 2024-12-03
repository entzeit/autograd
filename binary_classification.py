from autograd.neural_network import NeuralNetwork
from autograd.data_generator import DataGenerator
from autograd.tensor import Tensor
from autograd.loss import Loss
import random

seed = random.randint(0, 1000)
seed = 1337
print(f"Seed: {seed}")
def xor(x, y):
    return (x and not y) or (not x and y)
def binary_function1(a,b,c,d,e):
    return xor(a,b) and (c or not b) or xor(d, not e)
def binary_function2(a,b,d,e):
    return xor(not a,not b) and xor(d, not e)
def binary_function3(a,b,c,d,e):
    return a and b and c and d or e
X_train, Y_train, X_test, Y_test = DataGenerator([binary_function1, binary_function2, binary_function3], binary=True, seed=seed).generate(size=200)

input_size = X_train[0].shape[0]
layers = [(5, Tensor.relu), (10, Tensor.relu), (3, Tensor.sigmoid)]
loss_function = Loss.mse
lr = 0.3
epochs = 500
debug_steps = 10

model = NeuralNetwork(input_size, layers, lr, seed)
for epoch in range(epochs):
    Y_pred = model.forward(X_train, Y_train)
    loss = loss_function(Y_pred, Y_train)
    if ((epoch + 1) % (epochs/debug_steps) == 0) or epoch == 0:
            print(f"Epoch [{epoch + 1:>{len(str(abs(epochs)))}}/{epochs}]\tLoss: {loss.array[0][0]:.4f}")
    loss.backward()
    model.update()

model.test(X_test, Y_test, tolerance=0.01, binary=True)