import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from sklearn.metrics import classification_report, accuracy_score

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

def train():
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=64)
    model.save_weights("classifier.h5")

def load():
    model.load_weights("classifier.h5")

def predict(domain_names):
    # Tokenize domain names
    sequences = tokenizer.texts_to_sequences(domain_names)
    # Pad sequences
    X_test = pad_sequences(sequences, maxlen=max_length, padding='post')
    # Predict
    predictions = model.predict(X_test)
    return predictions

def normalize_predictions(predictions):
    for i in range(len(predictions)):
        if predictions[i][0] > 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    return predictions

def test():
    load()  # Make sure the model and tokenizer are loaded
    print("Testing model")
    # Load the test dataset
    test_df = pd.read_csv("dga_domains.csv", names=["host","domain","class","subclass"])
    test_df = test_df[["class", "domain"]]
    test_df['class'] = test_df['class'].replace({'legit': 1, 'dga': 0})

    # Predict classes for the test dataset
    predictions = predict(test_df['domain'].values)
    predictions = normalize_predictions(predictions)
    # Evaluate the predictions
    print("Classification Report:")
    print(classification_report(test_df['class'].values, predictions))
    print("Accuracy Score:")
    print(accuracy_score(test_df['class'].values, predictions))

#TRAINED
test()