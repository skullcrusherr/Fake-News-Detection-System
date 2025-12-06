import os
import re
import string
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_PATH = os.path.join(BASE_DIR, "dataset", "Fake.csv")
TRUE_PATH = os.path.join(BASE_DIR, "dataset", "True.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "saved_models", "distilbert_fake_news")


# -----------------------------
# 1. Optional light cleaning
# -----------------------------
def basic_clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data():
    print(f"Reading: {FAKE_PATH}")
    df_fake = pd.read_csv(FAKE_PATH)

    print(f"Reading: {TRUE_PATH}")
    df_true = pd.read_csv(TRUE_PATH)

    # Kaggle dataset usually has: title, text, subject, date
    def make_content(df):
        title = df.get("title", "")
        text = df.get("text", "")
        title = title.fillna("")
        text = text.fillna("")
        return (title + " " + text).astype(str)

    df_fake["content"] = make_content(df_fake)
    df_true["content"] = make_content(df_true)

    df_fake["label_str"] = "FAKE"
    df_true["label_str"] = "REAL"

    df = pd.concat(
        [df_fake[["content", "label_str"]],
         df_true[["content", "label_str"]]],
        ignore_index=True,
    )

    df = df.dropna(subset=["content", "label_str"])

    # Light cleaning (we DON'T remove too much punctuation,
    # because transformers can handle raw-ish text)
    df["content"] = df["content"].apply(basic_clean)

    # Map labels to integers: REAL = 0, FAKE = 1 (just a convention)
    label_map = {"REAL": 0, "FAKE": 1}
    df["label"] = df["label_str"].map(label_map)

    print("Label counts:")
    print(df["label_str"].value_counts())

    return df, label_map


def tokenize_function(examples, tokenizer, max_length=256):
    return tokenizer(
        examples["content"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def main():
    df, label_map = load_data()

    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df[["content", "label"]].reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df[["content", "label"]].reset_index(drop=True))

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["content"],
    )
    eval_tokenized = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["content"],
    )

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
    
        # NEW ARGUMENT NAMES:
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
    
        logging_steps=100,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.01,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

    # Save model + tokenizer
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
