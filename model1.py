import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from read_csv import *

vocab_size = 128  # Assuming ASCII
maxlen = 10  # Max length of input word

words = readdata()
X, y = generate_data(words, maxlen=maxlen)

model = tf.keras.Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),
    LSTM(units=128, return_sequences=True),
    LSTM(units=128),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def train():
    model.fit(X, y, batch_size=64, epochs=1)

def load():
    model.load_weights("predict_letter.h5")

#load()
train()

for word in readdata():
    word = word.split('.')[0]
    input_word = [ord(char) for char in word.lower()]
    input_word = pad_sequences([input_word], maxlen=maxlen)  
    predicted_index = np.argmax(model.predict(input_word))
    predicted_letter = chr(predicted_index)
    print("Predicted letter for '{}' is '{}'.".format(word, predicted_letter))

model.save_weights("predict_letter.h5")
