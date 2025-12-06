import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "fake_news_model.pkl")

def load_model():
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    model = load_model()

    # Get the class order used internally
    classes = model.classes_
    print("Classes order:", classes)

    while True:
        text = input("\nEnter a news text (or 'q' to quit): ")
        if text.lower() == "q":
            break

        pred = model.predict([text])[0]

        # If classifier supports predict_proba, show probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}
        else:
            prob_dict = "N/A (no predict_proba support)"

        print(f"\nPrediction: {pred}")
        print("Probabilities:", prob_dict)
