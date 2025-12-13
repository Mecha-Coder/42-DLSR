import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(self):
            
            # Model parameters
            self.n = 0              # features
            self.W = np.array([])   # weights
            self.b = np.array([])   # bias

            # Input data
            self.m = 0              # size of data
            self.k = []             # categories
            self.k_size             # category size

    # ============================================================================
    # Load Trained Co-efficients
    # ============================================================================

    def load_coefficients(self, _W: NDArray, _b: NDArray) -> None:
        """Load model with previously trained & saved coefficients (Weights, Bias)"""
        
        W_shape = _W.shape
        b_shape = _b.shape
        print(f"W = {W_shape} | b = {b_shape}")

        if (_W.size == 0 or _b.size == 0 or len(_W) != len(_b)):
            raise ValueError("Load failed. Got issue with the shape")
        
        self.W = _W
        self.b = _b
        self.k_size = W_shape[0]
        self.n = W_shape[1]
        
        print("Model loaded with co-efficients trained with: ")
        print(f"[{self.k_size}] categories and [{self.n}] features")
    
    # ============================================================================
    # Save Co-efficients
    # ============================================================================

    

    # ============================================================================
    # Train Model with Gradient Descent
    # ============================================================================

    def train_gradient_descent():
        pass


s1 = np.zeros((4,5))
s2 = np.zeros((4,1))
print(len(s1), len(s2))