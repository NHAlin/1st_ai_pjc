import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

#data = {
#   "text": [
#        "I love this product",
#        "This is amazing",
#        "Very happy with the service",
#        "I hate this",
#        "This is terrible",
#        "Very disappointed",
#        "Absolutely fantastic",
#        "Worst experience ever"
#    ],
#    "label": [
#        "positive", "positive", "positive",
#        "negative", "negative", "negative",
#        "positive", "negative"
#    ]
#}
df = pd.read_csv("data.csv")
#df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy check
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✅ Model Accuracy: {accuracy:.2f}")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained and saved!")
