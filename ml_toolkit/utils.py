from .constant import *
import pandas as pd
import sys

# =====================================================

def err_msg(e):
    print(f"{RED}ERROR: {RESET}{e}")

# =====================================================

def load_csv(filename: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(filename, index_col="Index")
        return data
    except Exception as e:
        msg = str(e)
        idx = msg.find("] ")
        err_msg(msg[idx + 2:])
        sys.exit(1)

# =====================================================


if __name__ == "__main__":
    print(load_csv("../dataset/dataset_test.csv"))
