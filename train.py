import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset (columns: subject, body, attachment, label)
df = pd.read_csv("email dataset.csv")

print("Columns in dataset:", df.columns)  # Debugging step

# Combine subject + body into one text feature
df["Text"] = df["Subject"].fillna("") + " " + df["Body"].fillna("")

X = df[["Text", "Attachment"]]  # Features: text + attachment
y = df["Label"]                # Target labels: ham, spam, phishing, support

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("Text", TfidfVectorizer(), "Text"),
    ("Attachment", OneHotEncoder(), ["Attachment"])
])

# Full model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MultinomialNB())
])

# Train and save
model.fit(X, y)
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully with 4 labels (ham, spam, phishing, support) and attachment feature.")
