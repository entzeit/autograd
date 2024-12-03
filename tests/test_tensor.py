import numpy as np
import math
import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autograd.tensor import Tensor

class TestFunctions(unittest.TestCase):
    
    def test_dot_product(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor([[1],[2],[3]])
        goal = Tensor([[14],[14]])
        assert goal == a @ b
        
    def test_dot_product_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor([[4],[5],[6]])
        c = a @ b
        c.backward()
        goal = Tensor([[4,5,6],[4,5,6]])
        assert goal == a.grad
        
    def test_add(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor([[4,5,6],[7,8,9]])
        goal = Tensor([[5,7,9],[8,10,12]])
        assert goal == a + b
        
    def test_add_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor([[4,5,6],[7,8,9]])
        c = a + b
        c.backward()
        goal = Tensor(np.ones_like(a.array))
        assert goal == a.grad
        
    def test_special_add_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = a + a
        c.backward()
        goal = Tensor(np.full_like(a.array, 2))
        assert goal == a.grad
        
    def test_add_with_value(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = 5
        c = a + b
        goal = Tensor([[6,7,8],[6,7,8]])
        assert goal == c
        
    def test_radd_with_value(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = 5
        c = b + a
        goal = Tensor([[6,7,8],[6,7,8]])
        assert goal == c
        
    def test_radd_with_value2(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = 5.0
        c = b + a
        goal = Tensor([[6,7,8],[6,7,8]])
        assert goal == c
        
    def test_radd_with_list(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = [[1,2,3],[1,2,3]]
        c = b + a
        goal = Tensor([[2,4,6],[2,4,6]])
        assert goal == c
        
    def test_radd_with_array(self):
        a = Tensor([[1,2,3],[1,2,3]])
        """
        NumpyArrays have undesired behaviour where the return a scalar instead of the array
        when functions like __rmul__ are called. Therefore necessary to call tolist()
        """
        b = np.array([[1,2,3],[1,2,3]]).tolist()
        c = b + a
        goal = Tensor([[2,4,6],[2,4,6]])
        assert goal == c
        
    def test_radd_with_value_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = 5
        c = b + a
        c.backward()
        goal = Tensor(np.ones_like(a.array))
        assert goal == a.grad
        
    def test_sigmoid(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = a.sigmoid()
        f = lambda x: 1 / (1 + math.exp(-x))
        goal = Tensor([[f(1),f(2),f(3)],[f(1),f(2),f(3)]])
        assert goal == c
        
    def test_sigmoid_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = a.sigmoid()
        c.backward()
        s = lambda x: 1 / (1 + math.exp(-x))
        f = lambda x: s(x) * (1 - s(x))
        goal = Tensor([[f(1),f(2),f(3)],[f(1),f(2),f(3)]])
        assert goal == a.grad
        
    def test_tanh(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = a.tanh()
        f = lambda x: 1 - (2 / (math.exp(2*x) + 1))
        goal = Tensor([[f(1),f(2),f(3)],[f(1),f(2),f(3)]])
        assert goal == c
        
    def test_tanh_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = a.tanh()
        c.backward()
        s = lambda x: 1 - (2 / (math.exp(2*x) + 1))
        f = lambda x: 1 - s(x)**2
        goal = Tensor([[f(1),f(2),f(3)],[f(1),f(2),f(3)]])
        assert goal == a.grad
        
    def test_relu(self):
        a = Tensor([[-1,0,2],[1,-1.5,3]])
        c = a.relu()
        goal = Tensor([[0,0,2],[1,0,3]])
        assert goal == c
        
    def test_relu_backward(self):
        a = Tensor([[-1,0,2],[1,-1.5,3]])
        c = a.relu()
        c.backward()
        goal = Tensor([[0,0,1],[1,0,1]])
        assert goal == a.grad
        
    def test_leaky_relu(self):
        a = Tensor([[-1,0,2],[1,-1.5,3]])
        c = a.leaky_relu()
        goal = Tensor([[-0.1,0,2],[1,-0.15,3]])
        assert goal == c 
        
    def test_leaky_relu_backward(self):
        a = Tensor([[-1,0,2],[1,-1.5,3]])
        c = a.leaky_relu()
        c.backward()
        goal = Tensor([[0.1,0.1,1],[1,0.1,1]])
        assert goal == a.grad 
    
    def test_full(self):
        w = Tensor([[-1,0,2],[1,-1.5,3]])
        x = Tensor([[1],[-1],[2]])
        b = Tensor([[1],[1.5]])
        z = w @ x + b
        out = z.sigmoid()
        f = lambda x: 1 / (1 + math.exp(-x))
        goal = Tensor([[f(4)],[f(10)]])
        assert goal == out
        
    def test_full_backward(self):
        w = Tensor([[-1,0,2],[1,-1.5,3]])
        x = Tensor([[1],[-1],[2]])
        b = Tensor([[1],[1.5]])
        z = w @ x + b
        out = z.sigmoid()
        out.backward()
        goal = Tensor([[1,-1,2],[1,-1,2]] * (out.array * (1 - out.array)))
        assert goal == w.grad
        
    def test_mul(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor(np.full_like(a.array, 2))
        c = a * b
        goal = Tensor([[2,4,6],[2,4,6]])
        assert goal == c
        
    def test_rmul_with_value(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = 2
        c = b * a
        goal = Tensor([[2,4,6],[2,4,6]])
        assert goal == c
        
    def test_rmul_with_array(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = np.full_like(a.array, 2).tolist()
        c = b * a
        goal = Tensor([[2,4,6],[2,4,6]])
        assert goal == c
    
    def test_mul_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor(np.full_like(a.array, 2))
        c = a * b
        c.backward()
        goal = Tensor(np.full_like(a.array, 2))
        assert goal == a.grad
        
    def test_neg(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = -a
        goal = Tensor([[-1,-2,-3],[-1,-2,-3]])
        assert goal == c
        
    def test_neg_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        c = -a
        c.backward()
        goal = Tensor([[-1,-1,-1],[-1,-1,-1]])
        assert goal == a.grad
        
    def test_sub(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor([[4,5,6],[7,8,9]])
        c = a - b
        goal = Tensor([[-3,-3,-3],[-6,-6,-6]])
        assert goal == c
        
    def test_sub_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor([[4,5,6],[7,8,9]])
        c = a - b
        c.backward()
        goal = Tensor(np.ones_like(a.array))
        assert goal == a.grad
        
    def test_rsub_with_value(self):
        a = 2
        b = Tensor([[4,5,6],[7,8,9]])
        c = a - b
        goal = Tensor([[-2,-3,-4],[-5,-6,-7]])
        assert goal == c
        
    def test_rsub_with_value_backward(self):
        a = 2
        b = Tensor([[4,5,6],[7,8,9]])
        c = a - b
        c.backward()
        goal = Tensor(np.full_like(b.array, -1))
        assert goal == b.grad
        
    def test_pow(self):
        a = Tensor([[1],[2],[3]])
        c = a**2
        goal = Tensor([[1],[4],[9]])
        assert goal == c
        
    def test_pow_backward(self):
        a = Tensor([[1],[2],[3]])
        c = a**2
        c.backward()
        goal = Tensor([[2],[4],[6]])
        assert goal == a.grad
        
    def test_truediv(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor(np.full_like(a.array, 2))
        c = a / b
        goal = Tensor([[0.5,1,1.5],[0.5,1,1.5]])
        assert goal == c
        
    def test_truediv_backward(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = Tensor(np.full_like(a.array, 2))
        c = a / b
        c.backward()
        goal = Tensor(np.full_like(a.array, .5))
        assert goal == a.grad
        
    def test_abs(self):
        a = Tensor([[1],[-1],[-2]])
        c = a.abs()
        goal = Tensor([[1],[1],[2]])
        assert goal == c
        
    def test_abs_backward(self):
        a = Tensor([[1],[-1],[-2]])
        c = a.abs()
        c.backward()
        goal = Tensor([[1],[-1],[-1]])
        assert goal == a.grad
        
    def test_log(self):
        a = Tensor([[1],[4],[2]])
        c = a.log()
        goal = Tensor([[math.log(1)],[math.log(4)],[math.log(2)]])
        assert goal == c
        
    def test_log_backward(self):
        a = Tensor([[1],[4],[2]])
        c = a.log()
        c.backward()
        goal = Tensor([[1],[1/4],[1/2]])
        assert goal == a.grad
        
    def test_full_with_loss_backward(self):
        w = Tensor([[-1,0,2],[1,-1.5,3]])
        x = Tensor([[1],[-1],[2]])
        y = Tensor([[1],[-1]])
        b = Tensor([[1],[1.5]])
        z = w @ x + b
        y_pred = z.sigmoid()
        loss = (y_pred - y) ** 2
        loss.backward()
        goal = Tensor(([[1,-1,2],[1,-1,2]] * (y_pred.array * (1 - y_pred.array))) * 2 * (y_pred.array - y.array))
        assert goal == w.grad
        
    def test_sum(self):
        batch = [Tensor([[1],[4],[2]]), Tensor([[1],[4],[2]])]
        c = sum(batch)
        goal = Tensor([[2],[8],[4]])
        assert goal == c
        
    def test_sum_backward(self):
        a = Tensor([[1],[4],[2]])
        batch = [a, a]
        c = sum(batch)
        c.backward()
        goal = Tensor([[2],[2],[2]])
        assert goal == a.grad
        
    def test_mean(self):
        batch = [Tensor([[1],[4],[2]]), Tensor([[1],[4],[2]])]
        c = Tensor.mean(batch)
        goal = Tensor([[1],[4],[2]])
        assert goal == c
    
    def test_batch_forward(self):
        w = Tensor([[1,0,1],[1,2,-1]])
        X = [[[1],[-1],[2]], [[1],[0],[1]]]
        b = Tensor([[-1],[1]])
        Y = [[[1],[1]], [[0],[0]]]
        Y_pred = [(w @ x + b).relu() for x in X]
        loss_batch = [(y - y_pred)**2 for y, y_pred in zip(Y,Y_pred)]
        loss = Tensor.mean(loss_batch)
        goal = Tensor([[1],[1]])
        assert goal == loss
        
    def test_batch_forward_backward(self):
        w = Tensor([[1,0,1],[1,2,-1]])
        X = [[[1],[-1],[2]], [[1],[0],[1]]]
        b = Tensor([[-1],[1]])
        Y = [[[1],[1]], [[0],[0]]]
        Y_pred = [(w @ x + b).relu() for x in X]
        loss_batch = [(y - y_pred)**2 for y, y_pred in zip(Y,Y_pred)]
        loss = Tensor.mean(loss_batch)
        loss.backward()
        goal = Tensor([[2,-1,3],[1,0,1]])
        assert goal == w.grad
        
    def test_reset_grad(self):
        w = Tensor([[1,0,1],[1,2,-1]])
        X = [[[1],[-1],[2]], [[1],[0],[1]]]
        b = Tensor([[-1],[1]])
        Y = [[[1],[1]], [[0],[0]]]
        Y_pred = [(w @ x + b).relu() for x in X]
        loss_batch = [(y - y_pred)**2 for y, y_pred in zip(Y,Y_pred)]
        loss = Tensor.mean(loss_batch)
        loss.backward()
        loss._reset_grad()
        assert np.allclose(np.zeros_like(w.array), w.grad)
        assert np.allclose(np.zeros_like(b.array),b.grad)
        
    def test_identity(self):
        a = Tensor([[1,2,3],[1,2,3]])
        b = a.identity()
        c = b * 2
        c.backward()
        assert b == a
        assert np.allclose(np.full_like(a.array, 2), a.grad)
        assert np.allclose(a.grad, b.grad)
        
    def test_special_case(self):
        a = Tensor([[1]])
        b = 1 + a * -1
        c = 1 - a
        assert b == c
    
    def test_special_case_backward(self):
        a = Tensor([[1]])
        b = 1 + a * -1
        b.backward()
        assert np.allclose(np.full_like(a.array, -1), a.grad)
        
    def test_special_case_backward2(self):
        a = Tensor([[1]])
        b = 1 - a
        b.backward()
        assert np.allclose(np.full_like(a.array, -1), a.grad)
        
    
if __name__ == "__main__":
    unittest.main()