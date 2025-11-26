import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset (columns: body, category)
df = pd.read_csv("email dataset.csv")

print("Columns in dataset:", df.columns)  # Debugging step

X = df['body']       # email text
y = df['category']   # ham, spam, phishing, support

# Create pipeline
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train and save
model.fit(X, y)
joblib.dump(model, 'model.pkl')

print("Model trained and saved successfully with 4 labels (ham, spam, phishing, support).")
