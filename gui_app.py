import tkinter as tk
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def analyze():
    text = entry.get()
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    result_label.config(text=f"Prediction: {prediction}")

root = tk.Tk()
root.title("AI Sentiment Analyzer")

tk.Label(root, text="Enter Text").pack()

entry = tk.Entry(root, width=40)
entry.pack()

tk.Button(root, text="Analyze", command=analyze).pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
