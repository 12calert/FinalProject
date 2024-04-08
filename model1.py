import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from read_csv import *

maxlen = 10

words = readdata()
X, y = generate_data(words, maxlen=maxlen)

class Model1():
    def __init__(self, vocab_size=128):
        self.model = tf.keras.Sequential([
            Embedding(input_dim=vocab_size, output_dim=32, input_length=maxlen),
            LSTM(units=128, return_sequences=True),
            Dropout(0.2),
            LSTM(units=128),
            Dense(vocab_size, activation='softmax')
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def train(self):
        self.model.fit(X, y, batch_size=64, epochs=1)
        self.model.save_weights("model1.h5")

    def load(self):
        self.model.load_weights("model1.h5")

    def generate(self, index_max=20):
        predicted = []
        index = 0
        for word in readdata():
            word = word.split('.')[0]
            input_word = [ord(char) for char in word.lower()]
            input_word = pad_sequences([input_word], maxlen=maxlen)  
            predicted_index = np.argmax(self.model.predict(input_word))
            predicted_letter = chr(predicted_index)
            predicted.append(word + predicted_letter)
            index += 1
            if index > index_max:
                break
        return predicted

