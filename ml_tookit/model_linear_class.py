import numpy as np
import pandas as pd

class LogisticRegression:
    
    def __init__(me):
        
        me.c: np.ndarray = None   # class
        me.w: np.ndarray  = None  # weights
        me.b: np.ndarray  = None  # bias

    #________________________________________________________

    # Load |or| Fit model
    def fit_model(me, x, y):
        m, n, k, c = me.init_train(x, y)

        w_all = np.zeros((n, k))
        b_all = np.zeros((k, 1))

        mean, std = me.get_stats(x, m)
        x_train = (x - mean) / std

        for i, cat in enumerate(c):
            print(f"Train for {cat}")
            y_train = np.array((y == cat).astype(np.uint8))

            w, b = me.std_grad_descent(x_train, y_train, m, n)

            print(w)
            print(b)

            w_all[:, i:i + 1] = w / std
            b_all[i, 0] = b - np.sum(w * mean / std)
        
        print("Training done 🎉")
        me.w = w_all
        me.b = b_all

    # Training methods
    def stochastic_grad_descent(me): pass  
    def std_grad_descent(me, x, y, m, n):
        alpha=0.1
        iter=500000

        w = np.zeros((n, 1))
        b = 0.0

        for i in range(iter):
            p = me.model(w, x, b)

            if (i): print(f"\033[{1}A", end='')
            print(f"Iteration: {i} | Loss: {me.loss_ft(p, y, m)}")

            step_w = alpha * (1/m) * np.dot(x, (p - y).T)
            step_b = alpha * (1/m) * np.sum(p - y)

            w -= step_w
            b -= step_b

            if (me.stop_loop(step_w, step_b, i, iter)): break
        
        return w, b
       
    # Training utils
    def init_train(me, x, y):
        '''
        n = features, m = data size, k = no. of class
        y shape (1 x m) | k > 1 | m > k
        x shape (n x m)
        n > 0
        '''

        if (y.shape[0] != 1 or
            x.shape[0] < 1 or
            x.shape[1] != y.shape[1]): 
            raise Exception("Incorrect shape for X and Y")

        m = x.shape[1]
        n = x.shape[0]

        c =  np.unique(y)
        k = len(c)

        if (k < 1): raise Exception("Only 1 class in Y")
        if (k > m): raise Exception("Not enough training data")
        
        print("Initializing training")
        print("=====================")
        print("Class      : ", c)
        print("Class no.  : ", k) 
        print("Feature no.: ", n)
        print("Sample no. : ", m)
        print("_____________________________________________________\n")

        me.c = c
        return m, n, k, c
    def get_stats(me, x, m):
        mean = np.sum(x, axis=1, keepdims=True) / m

        variance = np.sum((x - mean) ** 2, axis=1, keepdims=True) / m
        std = np.sqrt(variance)
        return mean, std
    def stop_loop(me, step_w, step_b, i, iter):

        if (np.all(np.abs(step_w) <= 1e-15) and np.abs(step_b) <= 1e-15):
            print(f"Converged 👍")

        elif (i == iter - 1):
            print("Didn't converge 😞")

        else: return False
        return True
    def loss_ft(me, p, y, m):
        return -(1/m) * np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    def model(me, w, x, b):
        z = np.dot(w.T, x) + b
        return 1 / (1 + np.exp(-z))
    
