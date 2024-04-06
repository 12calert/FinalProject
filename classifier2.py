import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D

print("Loading datasets...")
advers_df = pd.read_csv("advers.csv")
majestic_df = pd.read_csv("majestic_million.csv")

full_df = pd.read_csv("dga_domains_full.csv", names=["label", "source", "domain"])
full_df = full_df[["label", "domain"]]
full_df['label'] = full_df['label'].replace({'legit': 1, 'dga': 0})

# Tokenize domain names
domains = full_df['domain'].values
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(domains)
sequences = tokenizer.texts_to_sequences(domains)

# Pad sequences
max_length = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# Labels
labels = full_df['label'].values

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model parameters
max_features = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_size = 128

# Build the model
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=64)

model.save_weights("classifier.h5")