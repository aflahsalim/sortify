import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("email dataset.csv")

# Create pipeline
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train and save
model.fit(df['text'], df['label'])
joblib.dump(model, 'model.pkl')

print("Model saved successfully.")
