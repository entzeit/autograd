import numpy as np
import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autograd.loss import Loss
from autograd.tensor import Tensor

class TestFunctions(unittest.TestCase):
    def test_mse(self):
        Y_pred = [Tensor([[1],[2]]), Tensor([[1],[2]])]
        Y = [[[0],[0]], [[1],[2]]]
        loss = Loss().mse(Y_pred, Y)
        goal = Tensor([[0.5],[2]])
        assert goal == loss
        
    def test_mse_backward(self):
        a = Tensor([[1]])
        Y_pred = [a]
        Y = [[[0]]]
        loss = Loss().mse(Y_pred, Y)
        loss.backward()
        goal = np.array([[2]])
        assert np.allclose(goal, a.grad)
        
    def test_mse_backward2(self):
        a = Tensor([[1]])
        Y_pred = [a]
        Y = [[[1]]]
        loss = Loss().mse(Y_pred, Y)
        loss.backward()
        goal = np.array([[0]])
        assert np.allclose(goal, a.grad)
        
    def test_mae(self):
        Y_pred = [Tensor([[1],[2]]), Tensor([[1],[2]])]
        Y = [[[0],[0]], [[1],[2]]]
        loss = Loss().mae(Y_pred, Y)
        goal = Tensor([[0.5],[1]])
        assert goal == loss
        
    def test_mae_backward(self):
        a = Tensor([[1]])
        Y_pred = [a]
        Y = [[[0]]]
        loss = Loss().mae(Y_pred, Y)
        loss.backward()
        goal = np.array([[1]])
        assert np.allclose(goal, a.grad)
    
    def test_mae2(self):
        a = Tensor([[0]])
        Y_pred = [a]
        Y = [[[0]]]
        loss = Loss().mae(Y_pred, Y)
        goal = Tensor([[0]])
        assert goal == loss
        
    def test_mae3(self):
        a = Tensor([[1]])
        Y_pred = [a]
        Y = [[[1]]]
        loss = Loss().mae(Y_pred, Y)
        goal = Tensor([[0]])
        assert goal == loss
        
    def test_mae_backward2(self):
        a = Tensor([[0]])
        Y_pred = [a]
        b = [[0]]
        Y = [b]
        loss = Loss().mae(Y_pred, Y)
        loss.backward()
        goal = np.array([[-1]])
        assert np.allclose(goal, a.grad)
        
    def test_mae_backward3(self):
        a = Tensor([[1]])
        Y_pred = [a]
        Y = [[[1]]]
        loss = Loss().mae(Y_pred, Y)
        loss.backward()
        goal = np.array([[-1]])
        assert np.allclose(goal, a.grad)
        
    def test_binary_cross_entropy(self):
        Y_pred = [Tensor([[1],[1]])]
        Y = [Tensor([[1],[1]])]
        loss = Loss().binary_cross_entropy(Y_pred, Y)
        goal = [[0],[0]]
        assert goal == loss
        
    def test_binary_cross_entropy_backward(self):
        a = Tensor([[1]])
        Y_pred = [a]
        b = Tensor([[1]])
        Y = [b]
        loss = Loss().binary_cross_entropy(Y_pred, Y)
        loss.backward()
        goal = np.array([[-1]])
        assert np.allclose(goal, a.grad)
    
        
if __name__ == "__main__":
    unittest.main()