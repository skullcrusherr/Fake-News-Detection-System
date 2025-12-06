import os
import re
import pandas as pd
from typing import List

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
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "saved_models", "distilbert_fake_news_multi")

MODEL_NAME = "distilbert-base-uncased"


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_content_from_cols(df: pd.DataFrame, possible_cols: List[str]) -> pd.Series:
    """
    Try to build a content column by concatenating whatever text columns exist.
    e.g., ['title', 'text'] or ['content'].
    """
    cols = [c for c in possible_cols if c in df.columns]
    if not cols:
        raise ValueError(f"No suitable text columns found in df. Available: {df.columns}")
    s = df[cols[0]].fillna("").astype(str)
    for c in cols[1:]:
        s = s + " " + df[c].fillna("").astype(str)
    return s


def load_kaggle_fake_true(fake_path: str, true_path: str) -> pd.DataFrame:
    print(f"Loading Kaggle Fake/True: {fake_path}, {true_path}")
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    df_fake["content"] = make_content_from_cols(df_fake, ["title", "text", "content"])
    df_true["content"] = make_content_from_cols(df_true, ["title", "text", "content"])

    df_fake["label_str"] = "FAKE"
    df_true["label_str"] = "REAL"

    return pd.concat(
        [df_fake[["content", "label_str"]], df_true[["content", "label_str"]]],
        ignore_index=True,
    )


def load_fake_real_csv(path: str) -> pd.DataFrame:
    """
    Expect something like:
        - text/content column: 'text' or 'content' or 'article'
        - label: either 'FAKE'/'REAL' or 0/1 (where 1=fake, 0=real)
        - sometimes column is named 'label' or 'target'
    """
    print(f"Loading Fake_Real-style CSV: {path}")
    df = pd.read_csv(path)

    # figure out text column
    df["content"] = make_content_from_cols(df, ["text", "content", "article", "title"])

    # figure out label column name
    label_col = None
    if "label" in df.columns:
        label_col = "label"
    elif "target" in df.columns:
        label_col = "target"
    elif "label_str" in df.columns:
        label_col = "label_str"

    if label_col is None:
        raise ValueError(f"No label/target column found in {path}. Columns: {df.columns}")

    # normalize to label_str as 'FAKE'/'REAL'
    if label_col in ["label", "target"] and (
        df[label_col].dtype == "int64" or df[label_col].dtype == "float64"
    ):
        # assume 1 = FAKE, 0 = REAL
        df["label_str"] = df[label_col].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")
    else:
        # assume already string-ish labels
        df["label_str"] = df[label_col].astype(str).str.upper()

    return df[["content", "label_str"]]




def load_fake_real_pair(fake_path: str, real_path: str) -> pd.DataFrame:
    """
    For datasets that split fake/real into two files, like gossipcop_fake.csv / gossipcop_real.csv
    """
    print(f"Loading pair: {fake_path}, {real_path}")
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    df_fake["content"] = make_content_from_cols(df_fake, ["title", "text", "content", "article"])
    df_real["content"] = make_content_from_cols(df_real, ["title", "text", "content", "article"])

    df_fake["label_str"] = "FAKE"
    df_real["label_str"] = "REAL"

    return pd.concat(
        [df_fake[["content", "label_str"]], df_real[["content", "label_str"]]],
        ignore_index=True,
    )


def load_bharat_excel(path: str) -> pd.DataFrame:
    """
    Loader for BharatFakeNewsKosh Excel.

    We will use English translated text if available:
      - 'Eng_Trans_News_Body' and/or 'Eng_Trans_Statement'
    Fall back to:
      - 'News Body', 'Statement', 'Text'
    Label column is 'Label' (assume 1=FAKE, 0=REAL or use strings).
    """
    print(f"Loading BharatFakeNewsKosh Excel: {path}")
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"Could not read {path} as Excel: {e}")
        return pd.DataFrame(columns=["content", "label_str"])

    # Choose appropriate text columns in order of preference
    text_cols_priority = [
        "Eng_Trans_News_Body",
        "Eng_Trans_Statement",
        "News Body",
        "Statement",
        "Text",
    ]

    cols = [c for c in text_cols_priority if c in df.columns]
    if not cols:
        print(f"No suitable text column in Bharat dataset {path}; skipping.")
        return pd.DataFrame(columns=["content", "label_str"])

    # Use the helper to combine any that exist
    df["content"] = make_content_from_cols(df, cols)

    # Handle label
    if "Label" in df.columns:
        if df["Label"].dtype == "int64" or df["Label"].dtype == "float64":
            # assume 1 = FAKE, 0 = REAL
            df["label_str"] = df["Label"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")
        else:
            df["label_str"] = df["Label"].astype(str).str.upper()
    elif "label" in df.columns:
        if df["label"].dtype == "int64" or df["label"].dtype == "float64":
            df["label_str"] = df["label"].apply(lambda x: "FAKE" if int(x) == 1 else "REAL")
        else:
            df["label_str"] = df["label"].astype(str).str.upper()
    else:
        print(f"No Label/label column in Bharat dataset {path}; skipping.")
        return pd.DataFrame(columns=["content", "label_str"])

    return df[["content", "label_str"]]


def load_all_datasets() -> pd.DataFrame:
    dfs = []

    fake_path = os.path.join(DATA_DIR, "Fake.csv")
    true_path = os.path.join(DATA_DIR, "True.csv")
    if os.path.exists(fake_path) and os.path.exists(true_path):
        dfs.append(load_kaggle_fake_true(fake_path, true_path))

    fr_path = os.path.join(DATA_DIR, "Fake_Real.csv")
    if os.path.exists(fr_path):
        dfs.append(load_fake_real_csv(fr_path))

    gossip_fake = os.path.join(DATA_DIR, "gossipcop_fake.csv")
    gossip_real = os.path.join(DATA_DIR, "gossipcop_real.csv")
    if os.path.exists(gossip_fake) and os.path.exists(gossip_real):
        dfs.append(load_fake_real_pair(gossip_fake, gossip_real))

    pol_fake = os.path.join(DATA_DIR, "politifact_fake.csv")
    pol_real = os.path.join(DATA_DIR, "politifact_real.csv")
    if os.path.exists(pol_fake) and os.path.exists(pol_real):
        dfs.append(load_fake_real_pair(pol_fake, pol_real))

    news_path = os.path.join(DATA_DIR, "news_dataset.csv")
    if os.path.exists(news_path):
        dfs.append(load_fake_real_csv(news_path))

    # Bharat Excel (name as per your tree)
    bharat_path = os.path.join(DATA_DIR, "bharatfakenewskosh (3).xlsx")
    if os.path.exists(bharat_path):
        dfs.append(load_bharat_excel(bharat_path))

    if not dfs:
        raise RuntimeError("No datasets found in dataset/")

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["content", "label_str"])

    # Basic clean
    df["content"] = df["content"].apply(basic_clean)
    df["label_str"] = df["label_str"].str.upper()

    # Filter anything not FAKE/REAL
    df = df[df["label_str"].isin(["FAKE", "REAL"])]

    print("Combined dataset size:", len(df))
    print(df["label_str"].value_counts())

    # Map to int labels
    label_map = {"REAL": 0, "FAKE": 1}
    df["label"] = df["label_str"].map(label_map)

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
    df, label_map = load_all_datasets()

    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    train_dataset = Dataset.from_pandas(train_df[["content", "label"]].reset_index(drop=True))
    eval_dataset = Dataset.from_pandas(eval_df[["content", "label"]].reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",      # transformers 4.57 uses eval_strategy
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=200,
        num_train_epochs=4,         # can increase if you’re patient
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

    print("Starting training on combined datasets...")
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
