import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from ml_tookit.model_linear_class import LogisticRegression
from ml_tookit.utils import Describe

try: 
    data = pd.read_csv(sys.argv[1])
    model = LogisticRegression(sys.argv[2])
except Exception as e: 
    print(e)
    sys.exit(1)

courses = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 
            'Ancient Runes', 'History of Magic', 'Transfiguration']

feature = data[courses].copy()

# Avoid using median 
# feature = feature.fillna(feature.median())
t = Describe(feature)
for i, course in enumerate(courses):
    feature[course] = feature[course].fillna(t.p50[i])

x_test = np.array(feature).T

answer = model.predict(x_test)
df = pd.DataFrame({'Hogwarts House': answer})
df.index.name = "Index"
df.to_csv("house.csv", index=True)

print("Result saved 💾 in house.csv")

