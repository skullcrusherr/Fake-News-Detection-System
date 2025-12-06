import os
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models", "distilbert_fake_news_multi")

# Label mapping we used in training: 0 = REAL, 1 = FAKE
LABELS = ["REAL", "FAKE"]


print(f"Loading DistilBERT multi-dataset model from: {MODEL_DIR}")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval()  # inference mode


def predict_news(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Take raw news text and return:
      - predicted label: 'REAL' / 'FAKE' / 'UNSURE'
      - probability dict: {'REAL': p0, 'FAKE': p1}
    """

    if not text or not text.strip():
        return "UNKNOWN", {}

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    real_p = float(probs[0])  # index 0 = REAL
    fake_p = float(probs[1])  # index 1 = FAKE

    # Default "hard" label
    if fake_p >= 0.8:
        pred_label = "FAKE"
    elif real_p >= 0.8:
        pred_label = "REAL"
    else:
        pred_label = "UNSURE"

    prob_dict = {"REAL": real_p, "FAKE": fake_p}
    return pred_label, prob_dict

if __name__ == "__main__":
    print("DistilBERT Fake News Detector (multi-dataset)")
    print("Labels: REAL / FAKE")

    while True:
        text = input("\nEnter news text (or 'q' to quit): ")
        if text.lower() == "q":
            break

        label, probs = predict_news(text)
        print(f"\nPrediction: {label}")
        print("Probabilities:", probs)
