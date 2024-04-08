from model1 import Model1
from model2 import Model2
import ganmodel

classifier = train_classifier()

data = { "domain": predicted }
df = pd.DataFrame(data)
df["label"] = 0
X = np.array([extract_features(domain) for domain in df['domain']])
y = df['label'].values

print(df)
predictions = classifier.predict(X)
print(predictions)
print(classification_report(y, predictions))

data = { "domain": words[:20] }
df = pd.DataFrame(data)
df["label"] = 1
y = df['label'].values

X = np.array([extract_features(domain) for domain in df['domain']])
print(df)
predictions = classifier.predict(X)
print(predictions)
print(classification_report(y, predictions))
