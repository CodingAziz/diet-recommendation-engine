import pandas as pd

DATA_PATH = 'Data/dataset_uncompressed.csv'

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    return df