import tensorflow as tf
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from read_csv import *

maxlen = 10

words = readdata()
words = remove_dot( words )
X, y = generate_data(words, maxlen=maxlen)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
          tf.keras.layers.Dense(dff, activation='relu'),
          tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model=d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])

        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


class Model2():
    def __init__(self, d_model=32, num_heads=2, dff=128, vocab_size=128, ):
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=d_model, input_length=maxlen),
            PositionalEncoding(maxlen, d_model),
            TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff),
            tf.keras.layers.GlobalAveragePooling1D(),
            Dense(vocab_size, activation='softmax')
            ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def train(self):
        self.model.fit(X, y, batch_size=64, epochs=1)
        self.model.save_weights("model2.h5")

    def load(self):
        self.model.load_weights("model2.h5")

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

#model = Model2()
#model.train()