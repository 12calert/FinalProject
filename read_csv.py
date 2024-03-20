import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def readdata():
    pd.options.display.max_rows = 9999
    df = pd.read_csv('majestic_million.csv')
    return df["Domain"].tolist()

def generate_data(words, maxlen):
    X = []
    y = []
    for word in words:
        word = word.split('.')[0]
        indices = [ord(char) for char in word.lower()]
        X.append(indices[:-1])
        y.append(indices[-1])
    X_padded = pad_sequences(X, maxlen=maxlen)
    return np.array(X_padded), np.array(y)