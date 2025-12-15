import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from ml_tookit.model_linear_class import LogisticRegression

model = LogisticRegression()

data = pd.read_csv("data/dataset_train.csv")

courses = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 
            'Ancient Runes', 'History of Magic', 'Transfiguration']

feature = data[courses]
feature = feature.fillna(feature.median())

x_train = np.array(feature).T
y_train = np.array([data["Hogwarts House"]])

model.fit_model(x_train, y_train)
answer = model.predict(x_train)

validate = y_train.flatten()

