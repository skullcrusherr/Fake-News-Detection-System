import os
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models", "distilbert_fake_news")

# Our label mapping during training: 0 = REAL, 1 = FAKE
LABELS = ["REAL", "FAKE"]

# Load once at module import
print(f"Loading DistilBERT model from: {MODEL_DIR}")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval()  # inference mode


def predict_news(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Take a raw news string and return:
      - predicted label: 'REAL' or 'FAKE'
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

    pred_idx = int(probs.argmax())
    pred_label = LABELS[pred_idx]
    prob_dict = {label: float(prob) for label, prob in zip(LABELS, probs)}

    return pred_label, prob_dict


if __name__ == "__main__":
    # Simple CLI tester
    print("DistilBERT Fake News Detector (REAL/FAKE)")
    while True:
        text = input("\nEnter news text (or 'q' to quit): ")
        if text.lower() == "q":
            break

        label, probs = predict_news(text)
        print(f"Prediction: {label}")
        print("Probabilities:", probs)
