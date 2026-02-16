import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("🧠 AI Sentiment Analyzer")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter text: ")

    if text.lower() == "exit":
        break

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    print(f"Prediction: {prediction}\n")
