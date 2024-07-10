import numpy as np
import pandas as pd

def compute_acc_norm(acc):
    # acc_norm = np.sqrt(np.sum(acc**2, axis=1))
    acc_norm = np.linalg.norm(acc, axis=1)
    return acc_norm

def getSlice(df, t1):
    t2 = t1 + pd.Timedelta('5s')
    return df.loc[t1:t2]