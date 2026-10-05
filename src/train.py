"""
Train and compare several ML models on the Breast Cancer Wisconsin dataset.

Run from the project root:
    python src/train.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target  # target: 0 = malignant, 1 = benign
    return X, y, list(data.target_names)


def build_models():
    """Each model gets a StandardScaler so features are on the same scale."""
    def make(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Logistic Regression": make(LogisticRegression(max_iter=1000)),
        "KNN": make(KNeighborsClassifier(n_neighbors=5)),
        "SVM (RBF)": make(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        "Random Forest": make(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    }


def main():
    X, y, class_names = load_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    counts = y.value_counts().sort_index()
    print(f"Class counts: {class_names[0]}={counts[0]}, {class_names[1]}={counts[1]}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    fitted = {}
    for name, model in build_models().items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "CV Accuracy": cv.mean(),
            "Test Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
        fitted[name] = model

    table = pd.DataFrame(results).set_index("Model").round(4)
    table = table.sort_values("F1", ascending=False)
    print(table, "\n")
    table.to_csv(REPORTS_DIR / "model_comparison.csv")

    best_name = table.index[0]
    best = fitted[best_name]
    print(f"Best model: {best_name}\n")
    print(classification_report(y_test, best.predict(X_test), target_names=class_names))

    # Confusion matrix of the best model
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        best, X_test, y_test, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Model comparison bar chart
    ax = table[["CV Accuracy", "Test Accuracy", "F1"]].plot(
        kind="bar", figsize=(8, 4.5), ylim=(0.9, 1.0), rot=15
    )
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # Feature importance from a Random Forest (for interpretation)
    rf = fitted["Random Forest"].named_steps["clf"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    importances.sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Top 10 Important Features (Random Forest)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    joblib.dump(
        {"model": best, "feature_names": list(X.columns), "class_names": class_names},
        MODELS_DIR / "best_model.joblib",
    )
    print(f"\nSaved model to {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved plots/tables to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
