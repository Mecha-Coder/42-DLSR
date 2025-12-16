import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_tookit.utils import Describe
import pandas as pd

data = pd.read_csv("data/dataset_train.csv")
t = Describe(data)
t.show()