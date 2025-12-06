import os
import re
import string

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# -----------------------------
# 1. Basic text cleaning
# -----------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # remove numbers
    text = re.sub(r"\d+", "", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# 2. Load and combine Fake/True CSVs
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_PATH = os.path.join(BASE_DIR, "dataset", "Fake.csv")
TRUE_PATH = os.path.join(BASE_DIR, "dataset", "True.csv")

def load_data():
    print(f"Reading: {FAKE_PATH}")
    df_fake = pd.read_csv(FAKE_PATH)

    print(f"Reading: {TRUE_PATH}")
    df_true = pd.read_csv(TRUE_PATH)

    # The Kaggle dataset usually has: title, text, subject, date
    # We'll combine title + text into one 'content' column

    def make_content(df):
        title = df.get("title", "")
        text = df.get("text", "")
        title = title.fillna("")
        text = text.fillna("")
        return (title + " " + text).astype(str)

    df_fake["content"] = make_content(df_fake)
    df_true["content"] = make_content(df_true)

    # Add labels: FAKE / REAL
    df_fake["label"] = "FAKE"
    df_true["label"] = "REAL"

    # Keep only needed columns
    df = pd.concat([df_fake[["content", "label"]],
                    df_true[["content", "label"]]],
                   ignore_index=True)

    # Drop missing
    df = df.dropna(subset=["content", "label"])

    # Clean text
    print("Cleaning text...")
    df["clean_text"] = df["content"].apply(clean_text)

    X = df["clean_text"].values
    y = df["label"].values

    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())

    return X, y


# -----------------------------
# 3. Split data
# -----------------------------
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# -----------------------------
# 4. Build and train the model
# -----------------------------
def train_model(X_train, y_train):
    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,  # you can tune this later
                    ngram_range=(1, 2),  # unigrams + bigrams
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)
    return pipeline


# -----------------------------
# 5. Evaluate
# -----------------------------
def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))


# -----------------------------
# 6. Save model
# -----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")


# -----------------------------
# 7. Main
# -----------------------------
if __name__ == "__main__":
    print("🔹 Loading data...")
    X, y = load_data()

    print("\n🔹 Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\n🔹 Training...")
    model = train_model(X_train, y_train)

    print("\n🔹 Evaluation results:")
    evaluate_model(model, X_test, y_test)

    print("\n🔹 Saving model...")
    save_model(model)

    print("\n🎉 Done.")
