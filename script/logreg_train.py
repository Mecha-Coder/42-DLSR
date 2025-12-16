import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from ml_tookit.model_linear_class import LogisticRegression
from ml_tookit.utils import Describe, accuracy

model = LogisticRegression()

try: 
    data = pd.read_csv(sys.argv[1])
except Exception as e: 
    print(e)
    sys.exit(1)

courses = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 
            'Ancient Runes', 'History of Magic', 'Transfiguration']

feature = data[courses].copy()

t = Describe(feature)
for i, course in enumerate(courses):
    feature[course] = feature[course].fillna(t.p50[i])

x_train = np.array(feature).T
y_train = np.array([data["Hogwarts House"]])

model.fit_model(x_train, y_train)
model.save_model('logreg.npz')


# -----------------------------------------------
# Test

# y_pred = model.predict(x_train)

# accuracy(y_pred, y_train.flatten())  # 98.2%
