import numpy as np
import pandas as pd

def predict(w, x, b):
    z = np.dot(w.T, x) + b
    return 1 / (1 + np.exp(-z))

def train_GB(x, y, alpha=0.1, iter=10000000, limit=0.0000001):
    
    m = x.shape[1]
    n = x.shape[0]

    w = np.zeros((n, 1))
    b = 0

    for i in range(iter):
        p = predict(w, x, b)
        loss = -(1/m) * np.sum(y*np.log(p) + (1-y)*np.log(1 - p))

        step_w = alpha * (1/m) * np.dot(x, (p - y).T)
        step_b = alpha * (1/m) * np.sum(p - y)

        w -= step_w
        b -= step_b

        if (np.all(np.abs(step_w)) <= limit and np.abs(step_b) <= limit):
            print("Converge")
            break
        
    return w, b

def train(x, y):
    
    m = x.shape[1]
    n = x.shape[0]

    houses = sorted(list(set(y)))
    k = len(houses)

    mean = np.sum(x, axis=1, keepdims=True) / m
    variance = np.sum((x - mean) ** 2, axis=1, keepdims=True) / m
    std = np.sqrt(variance)

    x_train = (x- mean) / std

    w_all = np.zeros((n,k))
    b_all = np.zeros((k,1))


    for i, h in enumerate(houses):
        print(f"Train for {h} house")
        y_train = np.where(y == h, 1, 0).astype(np.uint8)
        y_train = y_train.reshape(1, m)

        w, b = train_GB(x_train, y_train)

        W_denorm = w / std
        b_denorm = b - np.sum(w * mean / std)
        print(w)
        print(b)
        print("---------")

        w_all[:, i:i + 1] = W_denorm
        b_all[i, 0] = b_denorm

    return w_all, b_all



data = pd.read_csv("data/dataset_train.csv")

courses = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 
            'Ancient Runes', 'History of Magic', 'Transfiguration']

x_train = data[courses]
x_train = np.array(x_train.fillna(x_train.median()))
x_train = x_train.T
x_test = x_train

y_train = data["Hogwarts House"]

W_all, b_all = train(x_train, y_train)

y_predict = predict(W_all, x_test, b_all)
answer = np.argmax(y_predict, axis=0)

houses = sorted(list(set(y_train)))
output = []
for i in answer:
    output.append(houses[i])

count = 0
for i,j in zip(y_train, output):
    if (i == j): count += 1

print(count / 1600 * 100)
