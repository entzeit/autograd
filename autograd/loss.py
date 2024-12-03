from autograd.tensor import Tensor
import numpy as np

class Loss():   
    """
    Required that Y_pred is [Tensor] and Y is [ [[]] ] or [ np.ndarray ]
    """
    
    @classmethod
    def mse(self, Y_pred, Y):
        result = []
        for y, y_pred in zip(Y,Y_pred):
            y = Tensor(y)
            out = (y - y_pred)**2
            result.append(out)
        return Tensor.mean(result)
    
    @classmethod
    def mae(self, Y_pred, Y):
        result = []
        for y, y_pred in zip(Y,Y_pred):
            y = Tensor(y)
            out = (y - y_pred).abs()
            result.append(out)
        return Tensor.mean(result)
   
    # FIXME: Buggy when used in neural network
    @classmethod
    def binary_cross_entropy(self, Y_pred, Y):
        result = []
        for y, y_pred in zip(Y,Y_pred): 
            y = Tensor(y)
            out = -((y * y_pred.log()) + ((1-y) * ((1-y_pred).log())))
            result.append(out)
        return Tensor.mean(result)