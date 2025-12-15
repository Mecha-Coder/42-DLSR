import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt

class Binary:

    def __init__(self):

        self.m: int = 0                # size of data    
        self.n: int = 0                # features
        self.W: np.ndarray = None      # weights
        self.b: float = 0.0            # bias
        self.cost: list[float] = None  # cost history for ploting

    #================================================================    

    def init_training(self, y: NDArray, X:NDArray):
        
        if (not np.all(np.isin(y, [0, 1])) or y.size < 2):
            raise Exception("Invalid input caterogy")
        if (X.size == 0 or y.size != X.shape[1]):
            raise Exception("Invalid feature dataset")

        self.n = X.shape[0]
        self.m = y.size
        self.W = np.zeros(self.n)
        self.b = 0
        self.cost = []

        print("Features  : ", self.n) 
        print("Data size : ", self.m)
        print("_____________________________________________________\n")

    #=======================================================================

    def get_stats(self, X: NDArray) -> tuple[NDArray, NDArray]: 
        """
            Compute mean and standard-deviation for every feature.
            Use for normalization and denormalization.
        """

        mean = np.sum(X, axis=1, keepdims=True) / self.m

        # Variance: Σ(x - μ)² / n
        variance = np.sum((X - mean) ** 2, axis=1, keepdims=True) / self.m
        std = np.sqrt(variance)

        return mean, std
    
    #=======================================================================

    def normalize(self, X: NDArray, mean: NDArray, std: NDArray)-> NDArray:
        """Normalize Z-Score, z = (x − μ) / σ"""
        return (X - mean) / std
    
    #=======================================================================

    def predict(self, X: NDArray) -> NDArray:
        """Predict the probability"""

        z = np.dot(self.W, X) + self.b
        return 1 / (1 + np.exp(-z))

    #=======================================================================

    def compute_loss(self, y_pred: NDArray, y: NDArray)-> float:
        """Compute the error where y = actual and y_pred is predicted"""
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # Add small epsilon to avoid log(0)
        
        total_error = np.sum(y * np.log(y_pred) + (1-y)*np.log(1-y_pred))
        return -(total_error /self.m) 

    #=====================================================================

    def gradient_W(self, y_pred: NDArray, y: NDArray, X: NDArray) -> NDArray:
        return np.dot(X, (y_pred - y).T) / self.m

    #=====================================================================

    def gradient_b(self, y_pred: NDArray, y: NDArray) -> float:
        return np.sum(y_pred - y) / self.m
    
    #=====================================================================

    def denormalize(self, mean: NDArray, std: NDArray):
        

        # W_orig = W_norm / σ
        W_denorm = self.W / std.reshape(len(std),)

        # b_orig = b_norm - W_norm · (μ / σ)
        b_denorm = self.b - np.sum(np.dot(self.W, mean / std))

        self.W = W_denorm
        self.b = b_denorm

    #=====================================================================

    def train_gradient_descent(self, y:NDArray, X:NDArray, learn_rate=0.2, iter=1000000, converge=0.00001) -> None:
        """Input binary caterogy and features (only support numerical)"""
        
        self.init_training(y, X)
        mean, std = self.get_stats(X)
        x_train = self.normalize(X, mean, std)

        for i in range(iter):
            if (i): print(f"\033[{1}A", end='')
            print(f"Iteration = {i}")

            y_predict = self.predict(x_train)
            loss = self.compute_loss(y_predict, y)
            self.cost.append(loss)

            gW = self.gradient_W(y_predict, y, x_train)
            gb = self.gradient_b(y_predict, y)

            self.W -= learn_rate * gW
            self.b -= learn_rate * gb

            if (np.all(np.abs(gW)) <= converge and abs(gb) <= converge):
                print(f"Converged 👍")
                break
            if (i == iter - 1):
                print("Training stopped. Didn't converge 😞")
        
        self.denormalize(mean, std)
        # print(self.W)
        # print(self.b)

# # Test ----------------------------------

def plot_loss(self):
    plt.plot(range(len(self.cost)), self.cost)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig("training_cost.png", dpi=300, bbox_inches="tight")
    plt.close()

data = pd.read_csv("./dataset/dataset_train.csv")

remove_columns = [
    "Index", "First Name", "Last Name", "Birthday", "Best Hand",
    "Potions", "Care of Magical Creatures", "Charms", "Flying",
    "Defense Against the Dark Arts","Arithmancy"
]
data.drop(columns=remove_columns, inplace=True)

for col in data.columns:
    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(data[col].median())

x_train = np.array(data.iloc[:, 1:])
x_train = x_train.T

y_train = np.where(np.array(data["Hogwarts House"]) == "Ravenclaw", 1, 0)


model = Binary()
model.train_gradient_descent(y_train, x_train)

x_test = np.array(data.iloc[0, 1:])
x_test = x_test.reshape(7,1)
print(model.predict(x_test))

# model.plot_loss()



    # def load_coefficients(self, W: NDArray, b: float) -> None:
    #     """Load model with previously trained & saved coefficients (Weights, Bias)"""
        
    #     if (W.size == 0 or b.size == 0 ):
    #         raise Exception("Invalid input")
        
    #     self.W = W
    #     self.b = b
    #     self.n = W.size

    #     print("Model loaded with trained co-efficients: ")
    #     print("Feature no : ", self.n)
    #     print("_____________________________________________________\n")


    # def convert_cat(self, k: list) -> NDArray:
    # """Convert categories into numerical representation. Using One-Hot Encoding method"""

    # lookup = {}
    # for i, cat in enumerate(self.k):
    #     lookup[cat] = i
    
    # one_hot = np.zeros((self.k_size, self.m), dtype=np.uint8)

    # for i, cat in enumerate(k):
    #     one_hot[lookup[cat], i] = 1

    # return one_hot