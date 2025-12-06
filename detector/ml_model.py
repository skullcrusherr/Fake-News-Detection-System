import os
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from django.conf import settings
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to the saved multi-dataset model
MODEL_DIR = os.path.join(settings.BASE_DIR, "saved_models", "distilbert_fake_news_multi")

# Label mapping: 0 = REAL, 1 = FAKE
LABELS = ["REAL", "FAKE"]

print(f"[ML] Loading DistilBERT model from: {MODEL_DIR}")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.eval()  # important for inference


def predict_news(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Takes raw news text and returns:
      - label: 'REAL', 'FAKE', or 'UNSURE'
      - probabilities: {'REAL': p0, 'FAKE': p1}
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

    # Simple safety/uncertainty rule
    if fake_p >= 0.8:
        label = "FAKE"
    elif real_p >= 0.8:
        label = "REAL"
    else:
        label = "UNSURE"

    prob_dict = {"REAL": real_p, "FAKE": fake_p}
    return label, prob_dict
