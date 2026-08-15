from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .preprocessor import create_vectorizer


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "training_data.csv"
MODEL_FILE = BASE_DIR / "models" / "text_classifier.joblib"


def load_data():
    # Load the labeled dataset used to train the classifier.
    df = pd.read_csv(DATA_FILE)

    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError(
            "Dataset must contain 'text' and 'category' columns."
        )

    return df


def train_model():
    # Prepare the dataset for training and evaluation.
    df = load_data()

    X = df["text"].astype(str)
    y = df["category"]

    # Keep some data unseen during training so the model can be evaluated fairly.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Convert text into numerical features and classify it with Logistic Regression.
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

    # Store the model's evaluation results so the application can display them.
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

    # Save the trained model together with its evaluation metrics.
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
    # Train a model if one does not already exist.
    if not MODEL_FILE.exists():
        model, _ = train_model()
        return model

    saved_model = joblib.load(MODEL_FILE)

    # Load the model from the new saved format.
    if isinstance(saved_model, dict) and "model" in saved_model:
        return saved_model["model"]

    # Support older model files if one still exists.
    return saved_model


def load_metrics():
    # Train the model if metrics do not exist yet.
    if not MODEL_FILE.exists():
        _, metrics = train_model()
        return metrics

    saved_model = joblib.load(MODEL_FILE)

    # Return the saved evaluation metrics.
    if isinstance(saved_model, dict) and "metrics" in saved_model:
        return saved_model["metrics"]

    # Rebuild the model using the new format if an old model file is found.
    _, metrics = train_model()
    return metrics


def classify_text(text):
    # Load the trained classifier and predict the category.
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