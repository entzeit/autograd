from autograd.tensor import Tensor
import numpy as np

class NeuralNetwork:   
    def __init__(self, input_size, layers, lr, seed=None):
        self.lr = lr
        self.layers = layers
        #initialize parameters
        self.W = []
        self.B = []
        self.input_size = input_size
        self._init_params(seed)
            
    def _init_params(self, seed):
        np.random.seed(seed)
        x_dim = self.input_size
        for layer in self.layers:
            y_dim = layer[0]
            self.W.append(Tensor(np.random.uniform(-1, 1, size=(y_dim, x_dim))))
            self.B.append(Tensor(np.random.uniform(-1, 1, size=(y_dim, 1))))
            x_dim = y_dim
        
    def forward(self, X, Y):
        if self.layers[-1][0] != Y[0].shape[0]:
            raise ValueError(f"Output layer and label mismatch")
        Y_pred = []
        for x in X:
            x = Tensor(x)
            for layer, w, b in zip(self.layers, self.W, self.B):
                activation = layer[1]
                y_pred = activation(w @ x + b)
                x = y_pred
            Y_pred.append(y_pred)
        return Y_pred
    
    def update(self):
        for i in range(len(self.W)):
            self.W[i].array -= self.W[i].grad * self.lr
            self.B[i].array -= self.B[i].grad * self.lr
            
    def __repr__(self):
        output = ""
        for i in range(len(self.W)):
            output += f"Layer {i}:\n"
            output += f"W: {self.W[i]}\n"
            output += f"B: {self.B[i]}\n"
        return output
    
    def test(self, X, Y, tolerance=0.001, binary=False):
        Y_pred = self.forward(X, Y)
        confusion_matrix = [[0,0],[0,0]]
        result = []
        for y_pred, y in zip(Y_pred, Y):
            for y_pred_row, y_row in zip(y_pred.array, y):
                y_pred_value = int(y_pred_row[0] > 0.5) if binary else y_pred_row[0]
                y_value = int(y_row[0] > 0.5) if binary else y_row[0]
                result.append(abs(y_pred_value - y_value) <= tolerance)
                if binary:
                    confusion_matrix[y_pred_value][y_value] += 1
        if binary:
            print('Confusion matrix:')
            print(confusion_matrix)
        accuracy = sum(result) / len(result)
        print(f'Accuracy on test data: {accuracy:.2f}')
        a = 5
        
        
        
    
