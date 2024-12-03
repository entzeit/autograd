import numpy as np
import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autograd.data_generator import DataGenerator

class TestFunctions(unittest.TestCase):
    def test_params(self):
        def func1(a, b):
            return float((a > 0.5) and (b > 0.5))
        def func2(c, d, e):
            return 0
        generator = DataGenerator([func1, func2, (lambda x: x + 1)])
        assert len(generator.params) == 6
    
    def test_init(self):
        def func1(a, b):
            return float((a > 0.5) and (b > 0.5))
        X_train, Y_train, X_test, Y_test = DataGenerator([func1, (lambda a: a + 1)]).generate(10, 0.7)
        assert len(X_train) == len(Y_train)
        assert len(X_train) == 7
        assert len(X_test) == 3
        
    def test_values(self):
        def func1(a, b, c, d, e, f, g, h):
            return 0
        X_train, Y_train, X_test, Y_train = DataGenerator([func1, (lambda a: a + 1)], 0, 1000).generate()
        assert np.all(X_train[0] >= 0)

if __name__ == "__main__":
    unittest.main()