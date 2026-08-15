from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from src.preprocessor import create_vectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "training_data.csv"
MODEL_FILE = BASE_DIR / "models" / "text_classifier.joblib"


def load_data():
    # Load the labeled examples used to train and evaluate the classifier.
    df = pd.read_csv(DATA_FILE)

    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'category' columns."
        )

    return df


def train_model():
    df = load_data()

    X = df["text"].astype(str)
    y = df["category"]

    # Keep part of the dataset unseen during training so we can measure
    # how well the model performs on messages it has not seen before.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # TF-IDF converts text into numerical features, while Logistic
    # Regression learns how those features relate to each category.
    model = Pipeline([
        ("tfidf", create_vectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "categories": sorted(y.unique().tolist()),
    }

    print(f"Model accuracy: {accuracy:.2%}")
    print()
    print(classification_report(y_test, predictions))

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save both the trained pipeline and its evaluation information so
    # the application can display the model's actual performance.
    joblib.dump(
        {
            "model": model,
            "metrics": metrics
        },
        MODEL_FILE
    )

    print(f"Model saved to: {MODEL_FILE}")

    return model, metrics


def load_model():
    if not MODEL_FILE.exists():
        model, _ = train_model()
        return model

    saved_model = joblib.load(MODEL_FILE)

    # Support the new saved format while remaining compatible with an
    # older model file if one still exists.
    if isinstance(saved_model, dict) and "model" in saved_model:
        return saved_model["model"]

    return saved_model


def load_metrics():
    if not MODEL_FILE.exists():
        _, metrics = train_model()
        return metrics

    saved_model = joblib.load(MODEL_FILE)

    if isinstance(saved_model, dict) and "metrics" in saved_model:
        return saved_model["metrics"]

    # If an older model file exists, retrain once to generate metrics.
    _, metrics = train_model()
    return metrics


def classify_text(text):
    model = load_model()

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    classes = model.classes_

    confidence = probabilities[
        list(classes).index(prediction)
    ]

    return {
        "category": prediction,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    train_model()