import numpy as np

def accuracy(predict, actual):
    """Correct prediction / total"""
    
    if len(predict) != len(actual):
        raise Exception("Incorrect shape")
    acc = (np.sum(predict == actual) / len(actual)) * 100
    print(f"Accuracy = {acc:.3}%")


    