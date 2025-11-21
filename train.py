import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Example dataset (replace with your own or expand it)
data = {
    "text": [
        "Congratulations, you won a prize!",
        "Reset your password immediately",
        "Server down, please fix ASAP",
        "Your invoice is attached",
        "Suspicious login attempt detected"
    ],
    "label": ["spam", "phishing", "support", "support", "phishing"]
}
df = pd.DataFrame(data)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Evaluate performance
print(classification_report(y, model.predict(X)))

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Training complete. Files saved: model.pkl, vectorizer.pkl")
