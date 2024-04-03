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

# Assign labels
advers_df['label'] = 0
majestic_df['label'] = 1

# Select relevant columns and limit the size
advers_df = advers_df[["domain", 'label']]
majestic_df = majestic_df[["domain", 'label']]

# Balancing the dataset for simplicity
advers_df = advers_df.iloc[:2000]
majestic_df = majestic_df.iloc[:2000]

# Combine and shuffle
combined_df = pd.concat([advers_df, majestic_df], ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# Tokenize domain names
domains = combined_df['domain'].values
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(domains)
sequences = tokenizer.texts_to_sequences(domains)

# Pad sequences
max_length = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_length, padding='post')

# Labels
labels = combined_df['label'].values

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# Model parameters
max_features = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_size = 128

# Build the model
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)