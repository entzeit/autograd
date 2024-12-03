import inspect
import random
import numpy as np

class DataGenerator:
    def __init__(self, functions, value_min=0, value_max=1, binary=True, seed=None):
        self.value_min = value_min
        self.value_max = value_max
        self.functions = functions
        self.binary = binary
        if seed is not None:
            random.seed(seed)
        self.params = self._get_params()
    
    def _get_params(self):
        parameter_set = set()
        for func in self.functions:
            sig = inspect.signature(func)
            parameter_set.update(sig.parameters.keys())
        return sorted(parameter_set)
    
    def generate(self, size=100, split=0.7, normalize=False):
        X = []
        Y = []
        train_size = int(split * size)
        for _ in range(size):
            x, y = self.call_functions()
            if normalize:
                x = self._normalize(x)
                y = self._normalize(y)
            X.append(x)
            Y.append(y)
        X_train, X_test = X[:train_size], X[train_size:]
        Y_train, Y_test = Y[:train_size], Y[train_size:]
        return (X_train, Y_train, X_test, Y_test)
    
    def call_functions(self):
        results = []
        parameter_values = self._generate_inputs()
        for func in self.functions:
            func_params = inspect.signature(func).parameters.keys()
            func_args = {param: parameter_values[param] for param in func_params}
            results.append(float(func(**func_args)))
        return self._format(parameter_values, results)
    
    def _generate_inputs(self):
        if self.binary:
            return {param: random.randint(0, 1) for param in self.params}
        return {param: random.uniform(self.value_min, self.value_max) for param in self.params}

    def _format(self, parameter_values, results):
        return np.array(list(parameter_values.values())).reshape(-1, 1), \
            np.array(results).reshape(-1, 1)
    
    def _normalize(self, value):
        if self.value_max == self.value_min:
            raise ValueError("max_value and min_value cannot be the same.")
        return 2 * (value - self.value_min) / (self.value_max - self.value_min) - 1
