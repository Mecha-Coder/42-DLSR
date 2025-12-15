import numpy as np
import pandas as pd
from numpy.typing import NDArray



class LogisticR:

    def __init__(self):

        self.m: int = 0                # size of data   
        self.n: int = 0                # features

        self.k: int = 0                # number of categories
        self.k_list: list[str] = None  # category list
        
        self.w: np.ndarray = None      # weights
        self.b: np.ndarray = None      # bias
  

    # ==================================================================

    def model(self, w, x, b):

        z = np.dot(w.T, x) + b
        return 1 / (1 + np.exp(-z))
    
    # ==================================================================

    def get_stats(self, x):
        mean = np.sum(x, axis=1, keepdims=True) / self.m

        variance = np.sum((x - mean) ** 2, axis=1, keepdims=True) / self.m
        std = np.sqrt(variance)

        return mean, std
    
    # ==================================================================
    
    def compute_loss(self, p, y):
        m = self.m
        return -(1/m) * np.sum(y * np.log(p) + (1-y) * np.log(1-p))

    # ==================================================================

    def train_GD(self, x, y, alpha=0.1, iter=9000000, limit=0.000001):
                
        w = np.zeros((self.n, 1))
        b = 0
        m = self.m

        for i in range(iter):

            p = self.model(w, x, b)
            loss = self.compute_loss(p, y)
            
            if (i): print(f"\033[{1}A", end='')
            print(f"Iteration: {i} | Loss: {loss}")

            dw = (1/m) * np.dot(p - y, x.T)
            step_w = alpha * dw.T
            step_b = alpha * (1/m) * np.sum(p - y)

            w -= step_w
            b -= step_b

            if (np.all(np.abs(step_w) <= limit) and np.abs(step_b) <= limit):
                print("Converged 👍")
                break
            if (i == iter - 1):
                print("Did not converge 🤔")
        
        return w, b

    # ==================================================================

    def init_train(self, x, y):

        self.k_list = sorted(list(set(y)))
        self.k = len(self.k_list)
        self.m = x.shape[1]
        self.n = x.shape[0]

        return self.m, self.n, self.k

    # ==================================================================

    def fit_model(self, x, y):
        
        m, n, k = self.init_train(x, y)
        mean, std = self.get_stats(x)
        x_train = (x - mean) / std

        w_all = np.zeros((n, k))
        b_all = np.zeros((k, 1))

        for i, cat in enumerate(self.k_list):
            print(f"Train for category: {cat}")
            y_train = (y == cat).astype(np.uint8)
            y_train =  y_train.reshape((1, m))

            w, b = self.train_GD(x_train, y_train)

            w_all[:, i:i + 1] = w / std
            b_all[i, 0] = b - np.sum(w * mean / std)

        self.w = w_all
        self.b = b_all


data = pd.read_csv("dataset/dataset_train.csv")

courses = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 
            'Ancient Runes', 'History of Magic', 'Transfiguration']

x_train = data[courses]
x_train = np.array(x_train.fillna(x_train.median()))
x_train = x_train.T

y_train = np.array(data["Hogwarts House"])

model = LogisticR()
model.fit_model(x_train, y_train)