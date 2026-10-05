"""
Load the trained model and predict on a sample.

Run from the project root (after training):
    python src/predict.py            # uses a random sample from the dataset
    python src/predict.py --index 10 # uses row 10 of the dataset
"""
import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Breast cancer prediction demo")
    parser.add_argument("--index", type=int, default=None, help="row number of the dataset")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python src/train.py` first.")

    bundle = joblib.load(MODEL_PATH)
    model, class_names = bundle["model"], bundle["class_names"]

    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    row = X.sample(1, random_state=None) if args.index is None else X.iloc[[args.index]]

    proba = model.predict_proba(row)[0]
    pred = int(model.predict(row)[0])
    actual = int(y.loc[row.index[0]])

    print(f"Sample row index : {row.index[0]}")
    print(f"Predicted        : {class_names[pred]} (confidence {proba[pred]:.1%})")
    print(f"Actual           : {class_names[actual]}")
    print("\nThis is a learning project, not a medical diagnosis tool.")


if __name__ == "__main__":
    main()
