import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Literal

class LogisticRegression:
    
    def __init__(me, file: str = None):
        
        me.c: np.ndarray = None   # class
        me.w: np.ndarray = None  # weights
        me.b: np.ndarray = None  # bias

        me.gradient_descent = {
            "batch": me.batch_grad_descent,
            "mini-batch": me.mini_batch_grad_descent,
            "stochastic": me.stochastic_grad_descent
        }

        if (file):
            data = np.load(file, allow_pickle=True)
            me.c = data['c']
            me.w = data['w']
            me.b = data['b']
            print("Model loaded")
            print("=============")
            print("Class      : ", me.c)
            print("Class no.  : ", me.b.shape[0]) 
            print("Feature no.: ", me.w.shape[0])
            print("_____________________________________________________\n")

    #________________________________________________________

    # Load |or| Fit model
    def fit_model(me, x, y , which: Literal["stochastic", "mini-batch", "batch"]):
        m, n, k, c = me.init_train(x, y)

        w_all = np.zeros((n, k))
        b_all = np.zeros((k, 1))

        mean, std = me.get_stats(x, m)
        x_train = (x - mean) / std

        fig, ax = plt.subplots(figsize=(10, 6))
        
        print(f"Gradient Descent Type: {which}\n")
        t_start = time.time()
        for i, cat in enumerate(c):
            print(f"> Train for {cat}")
            y_train = np.array((y == cat).astype(np.uint8))

            w, b ,loss = me.gradient_descent[which](x_train, y_train, m, n)
            ax.plot(range(len(loss)), loss, label=cat)
            
            w_all[:, i:i + 1] = w / std
            b_all[i, 0] = b - np.sum(w * mean / std)
        
        t_end = time.time()
        print(f"Training done 🎉, Time:{(t_end - t_start):.3f} s")
        me.w = w_all
        me.b = b_all
        return fig, ax
    def save_model(me, filepath):
        np.savez(filepath, c=me.c, w=me.w, b=me.b)
        print("Model saved ✌🤩")

    # Predict class
    def predict(me, x):
        if (me.w is None or me.b is None or me.c is None):
            return print("Please load or fit the model first")

        if (x.shape[0] != me.w.shape[0] and x.shape[1] < 1):
            raise Exception("Incorrect shape for X")
        
        p = me.model(x)
        idx_arr = np.argmax(p, axis=0)
        return me.c[idx_arr].flatten()

    # Training methods
    def stochastic_grad_descent(me, x, y, m, n):
        alpha=0.0001
        epoch = 10000

        loss = []
        w = np.zeros((n, 1))
        b = 0.0

        for i in range(epoch):
            ids = np.random.permutation(m)
            y1 = None
            p = None

            for r in ids:
                x1 = x[:, r:r+1]
                y1 = y[0, r]

                p = me.model(x1, w, b).item()                
                step_w = alpha * x1 * (p - y1)
                step_b = alpha * (p - y1)

                w -= step_w
                b -= step_b

                if (me.stop_loop(step_w, step_b, i, epoch)):
                    return w, b, loss
                
            loss.append(me.loss_ft(p, y1, 1))
            if (i): print(f"\033[{1}A", end='')
            print(f"Epoch: {i} | Loss: {loss[i]}")
    def mini_batch_grad_descent(me, x, y, m, n):
        alpha=0.0001
        epoch=10000
        batch_size = 32

        loss = []
        w = np.zeros((n, 1))
        b = 0.0

        for i in range(epoch):
            ids = np.random.permutation(m)
            ymini = None
            p = None

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                batch = ids[start:end]

                xmini = x[:, batch]
                ymini = y[:, batch]
                l = len(batch)

                p = me.model(xmini, w, b)
                step_w = alpha * (1/l) * np.dot(xmini, (p - ymini).T)
                step_b = alpha * (1/l) * np.sum(p - ymini)

                w -= step_w
                b -= step_b

                if (me.stop_loop(step_w, step_b, i, epoch)): 
                    return w, b, loss
            
            loss.append(me.loss_ft(p, ymini, l))
            if (i): print(f"\033[{1}A", end='')
            print(f"Epoch: {i} | Loss: {loss[i]}")
    def batch_grad_descent(me, x, y, m, n):
        alpha=0.001
        epoch=1000000

        loss = []
        w = np.zeros((n, 1))
        b = 0.0

        for i in range(epoch):
            p = me.model(x, w, b)
            loss.append(me.loss_ft(p, y, m)) 

            step_w = alpha * (1/m) * np.dot(x, (p - y).T)
            step_b = alpha * (1/m) * np.sum(p - y)

            w -= step_w
            b -= step_b

            if (me.stop_loop(step_w, step_b, i, epoch)): 
                return w, b, loss

            if (i): print(f"\033[{1}A", end='')
            print(f"Epoch: {i} | Loss: {loss[i]}")
       
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
        eps = 5e-6
        if (np.all(np.abs(step_w) <= eps) and np.abs(step_b) <= eps):
            print(f"Converged 👍")

        elif (i == iter - 1):
            print("Didn't converge 😞")

        else: return False
        print("_______________________________")
        return True
    def loss_ft(me, p, y, m):
        return -(1/m) * np.sum(y*np.log(p) + (1-y)*np.log(1-p))
    def model(me, x, w=None, b=None):

        if (w is None and b is None):
            w = me.w
            b = me.b
        z = np.dot(w.T, x) + b
        return 1 / (1 + np.exp(-z))
    
