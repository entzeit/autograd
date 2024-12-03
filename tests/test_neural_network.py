import numpy as np
import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autograd.neural_network import NeuralNetwork
from autograd.data_generator import DataGenerator
from autograd.tensor import Tensor
from autograd.loss import Loss

class TestFunctions(unittest.TestCase):
    def test_init(self):
        model = NeuralNetwork(2, [(1, Tensor.identity)], 0)
        assert model.W[0].array.shape == (1,2)
        assert model.B[0].array.shape == (1,1)
        
    def test_init_complex(self):
        model = NeuralNetwork(4, [(10, Tensor.identity), (2, None)], 0)
        assert model.W[0].array.shape == (10,4)
        assert model.B[0].array.shape == (10,1)
        assert model.W[1].array.shape == (2,10)
        assert model.B[1].array.shape == (2,1)
        
    def test_forward(self):
        X_train, Y_train, X_test, Y_test = DataGenerator([lambda x: x]).generate()
        model = NeuralNetwork(1, [(1, Tensor.identity)], 0)
        Y_pred = model.forward(X_train, Y_train)
        loss = Loss.mse(Y_pred, Y_train)
        assert (loss.array > 0)
        
    def test_forward_complex(self):
        X_train, Y_train, X_test, Y_test = DataGenerator([lambda x, y: x + y]).generate()
        model = NeuralNetwork(2, [(2, Tensor.sigmoid), (100, Tensor.relu), (1, Tensor.identity)], 0)
        Y_pred = model.forward(X_train, Y_train)
        loss = Loss.mse(Y_pred, Y_train)
        assert (loss.array > 0)
        
    def test_forward_complex_backward(self):
        X_train, Y_train, X_test, Y_test = DataGenerator([lambda x, y: x + y]).generate()
        model = NeuralNetwork(2, [(2, Tensor.sigmoid), (100, Tensor.relu), (1, Tensor.identity)], 0.01)
        Y_pred = model.forward(X_train, Y_train)
        loss = Loss.mse(Y_pred, Y_train)
        loss.backward()
        model.update()
        Y_pred2 = model.forward(X_train, Y_train)
        loss_after_update = Loss.mse(Y_pred2, Y_train)
        assert (loss.array > loss_after_update.array)
        
    def test_forward_complex_backward2(self):
        X_train, Y_train, X_test, Y_test = DataGenerator([(lambda x, y: x + y), (lambda x: -1 * x)]).generate()
        model = NeuralNetwork(2, [(2, Tensor.sigmoid), (100, Tensor.relu), (2, Tensor.identity)], lr=0.001, seed=1337)
        Y_pred = model.forward(X_train, Y_train)
        loss = Loss.mse(Y_pred, Y_train)
        loss.backward()
        model.update()
        Y_pred2 = model.forward(X_train, Y_train)
        loss_after_update = Loss.mse(Y_pred2, Y_train)
        assert np.all(loss.array > loss_after_update.array)
    
if __name__ == "__main__":
    unittest.main()