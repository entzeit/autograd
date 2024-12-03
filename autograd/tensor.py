import numpy as np
from queue import Queue

"""
This class can interact with other Tensors and all built in python classes.
Don't interact with NumPy arrays directly.
"""
class Tensor:
    def __new__(cls, data, *args, **kwargs):
        #prevent wrapping Tensor into Tensor:
        if isinstance(data, Tensor):
            return data
        return super().__new__(cls)
    
    def __init__(self, data, _prev=()):
        if isinstance(data, Tensor):
            return
        if isinstance(data, np.ndarray):
            self.array = data.astype(np.double)
        else:
            self.array = np.asarray(data, dtype=np.double)
        if self.array.ndim != 2:
            raise ValueError(f"2-dim matrix required for Tensor")
        self.grad = np.zeros_like(self.array)
        self._backward = lambda: None
        self._prev = _prev

    def __str__(self):
        return f"Tensor(array={self.array}, grad={self.grad})"
    
    __repr__ = __str__
       
    # --- Forward Pass Calculations ---
    # These functions are required for the first part of the forward pass
    # w @ x + b
    
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.array.shape[1] != other.array.shape[0]:
            raise ValueError(f"Dot product shape mismatch")
        out = Tensor(self.array @ other.array, (self, other))
        def _backward():
            """
            These operations need to be on the Tensor.array itself, 
            so they do not contribute to the autograd calculation graph
            while doing the backward pass
            """
            self.grad += out.grad @ other.array.T
            other.grad += self.array.T @ out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        if isinstance(other, (float, int)): 
            other = np.full_like(self.array, other)  #required for sum()
        other = other if isinstance(other, Tensor) else Tensor(other) #required for sum()
        if self.array.shape != other.array.shape:
            raise ValueError(f"Shape mismatch")
        out = Tensor(self.array + other.array, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    # --- Activation Calculations ---
    # These functions are required for the activation step in the forward pass
    
    def identity(self):
        return self

    def sigmoid(self):
        f = lambda x: 1 / (1 + np.exp(-x))
        out = Tensor(np.vectorize(f)(self.array), (self,)) #(self, ) needs to be tupel for iteration
        def _backward():
            self.grad += (out.array * (1 - out.array)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        f = lambda x: 1 - (2 / (np.exp(2*x) + 1))
        out = Tensor(np.vectorize(f)(self.array), (self,))
        def _backward():
            self.grad += (1 - out.array**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.where(self.array > 0, self.array, 0), (self,))
        def _backward():
            self.grad += np.where(self.array > 0, 1, 0) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self):
        slope = 0.1
        out = Tensor(np.where(self.array > 0, self.array, slope * self.array), (self,))
        def _backward():
            self.grad += np.where(self.array > 0, 1, slope) * out.grad 
        out._backward = _backward
        return out
    
    # --- Loss Calculations ---
    # These functions are required for the loss calculation
    
    # --- MSE ---
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            other = np.full_like(self.array, other)
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.array.shape != other.array.shape:
            raise ValueError(f"Shape mismatch")
        out = Tensor(self.array * other.array, (self, other))
        def _backward():
            self.grad += other.array * out.grad
            other.grad += self.array * out.grad
            a = 0
        out._backward = _backward
        return out
    
    # Redirection:
    # defining _backward() not necessary because gets split to subcalculations 
    # which does not operate on the self.array but on the Tensor which will define _backward()
    def __neg__(self):
        return self * (-1)
    
    # Redirection:
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.array**other, (self,))
        def _backward():
            self.grad += (other * self.array**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def mean(self):
        return sum(self) / len(self)
    
    # --- MAE Calculations ---
    def abs(self):
        out = Tensor(np.abs(self.array), (self,))
        def _backward():
            self.grad += np.where(self.array >= 0, 1, -1) * out.grad
        out._backward = _backward
        return out
    
    # --- Binary Cross-Entropy Calculations ---

    #__log__ does not exist
    def log(self):
        if not np.all(self.array >= 0):
            raise ValueError(f"Log only possible for positive values")
        #prevent infinity:
        epsilon=1e-15
        self.array[self.array == 0.0] = epsilon
        
        out = Tensor(np.log(self.array))
        def _backward():
            self.grad += self.array**(-1) * out.grad
        out._backward = _backward
        return out
    
    # --- Python Functions ---
    # Required for calling Python functions like sum() on Tensors,
    # because sum()'s internal first parameter is 0 (the default initial value).
    # Therefore Tensor is called via __radd__()
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self**-1
    
    # --- Backward Pass ---
    def backward(self):
        self._reset_grad()
        self.grad = np.ones_like(self.grad)
        queue = Queue()
        queue.put(self)
        while (not queue.empty()):
            tensor = queue.get()
            tensor._backward()
            for p in tensor._prev:
                queue.put(p)
    
    def _reset_grad(self):
        self.grad = np.zeros_like(self.array)
        for p in self._prev:
            p._reset_grad()
            
    # --- Unit Tests ---        
    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.array.shape != other.array.shape:
            raise ValueError(f"Shape mismatch")
        return np.allclose(self.array, other.array)
    
    def __req__(self, other):
        return other == self
        
        
