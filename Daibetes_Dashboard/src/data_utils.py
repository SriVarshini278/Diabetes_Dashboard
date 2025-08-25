import os
import pandas as pd

def load_data(path=None):
    if path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        path = os.path.join(base_dir, '..', 'data', 'diabetes.csv')
    path = os.path.normpath(path)
    try:
        return pd.read_csv(path, encoding='utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find dataset at: {path}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Empty dataset found at: {path}")


