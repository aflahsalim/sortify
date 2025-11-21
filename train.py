import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Sample training data
data = {
    "text": [
        "Congratulations! You've won a prize!",
        "Get rich quick with crypto!",
        "Your account has been suspended",
        "Hi team, meeting at 10 AM",
        "Thanks for your feedback",
        "Your order has been shipped"
    ],
    "label": ["spam", "spam", "spam", "ham", "ham", "ham"]
}

df = pd.DataFrame(data)

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved.")
