import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

def extract_features(domain):
    return np.array([len(domain), sum(c.isdigit() for c in domain), sum(c.isalpha() for c in domain)])

def train_classifier():

    print("Loading datasets...")
    advers_df = pd.read_csv("advers.csv")
    majestic_df = pd.read_csv("majestic_million.csv")

    advers_df['label'] = 0
    majestic_df['label'] = 1

    advers_df = advers_df[["domain", 'label']]
    majestic_df = majestic_df[["domain", 'label']]

    advers_df = advers_df.iloc[:2000]
    majestic_df = majestic_df.iloc[:2000]

    combined_df = pd.concat([advers_df, majestic_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    X = np.array([extract_features(domain) for domain in combined_df['domain']])
    y = combined_df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    #predictions = knn.predict(X_test)
    #print(classification_report(y_test, predictions))
    return knn