# Breast Cancer Detection using Machine Learning

A beginner-friendly, end-to-end ML project in Python. It trains and compares four
classifiers to predict whether a tumour is **malignant** or **benign** from 30
measurements of cell nuclei (Breast Cancer Wisconsin dataset, built into scikit-learn,
so no download is needed).

> Educational project only. Not a medical diagnosis tool.

## What it covers
- Data loading and class balance check
- Preprocessing with `StandardScaler` inside a `Pipeline` (prevents data leakage)
- Models: Logistic Regression, KNN, SVM (RBF), Random Forest
- 5-fold cross-validation + held-out test set (stratified 80/20 split)
- Metrics: accuracy, precision, recall, F1, confusion matrix
- Feature importance for interpretation
- Saving and loading the best model with `joblib`

## Project structure
```
breast-cancer-ml/
├── src/
│   ├── train.py      # train, compare, plot, save best model
│   └── predict.py    # load model and predict on a sample
├── models/           # saved model (created by train.py)
├── reports/          # plots and comparison table (created by train.py)
├── requirements.txt
└── README.md
```

## How to run
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py              # trains models, saves plots + model
python src/predict.py --index 10 # predict on dataset row 10
```

## Results (test set, 114 samples)
| Model | CV Acc | Test Acc | F1 |
|---|---|---|---|
| Logistic Regression | 0.980 | 0.983 | 0.986 |
| SVM (RBF) | 0.971 | 0.983 | 0.986 |
| KNN | 0.967 | 0.956 | 0.966 |
| Random Forest | 0.958 | 0.956 | 0.966 |

Plots are saved in `reports/`.

## Ideas to extend it (good for viva / report)
1. Hyperparameter tuning with `GridSearchCV`
2. Add ROC curve and AUC
3. Try XGBoost or a small neural network (Keras / PyTorch)
4. Build a web UI with Streamlit where users enter measurements
5. Use SHAP for explainability
6. Replace the dataset with a CSV of your own (e.g. diabetes, heart disease)

## Tech stack
Python, NumPy, pandas, scikit-learn, Matplotlib, joblib
