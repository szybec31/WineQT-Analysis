import pandas as pd
import numpy as np
from src.data_loader import load_data
from sklearn.model_selection import (
    cross_validate,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import sys

sys.stdout = open("Test_results.txt", "w", encoding="utf-8")

pd.set_option("display.max_columns", None)
df = load_data("Dataset/WineQT.csv")
df = df.drop("Id", axis=1)




# Zad 1 - Walidacja krzyżowa dla regresji liniowej i logistycznej

X_reg = df.drop("quality", axis=1)
y_reg = df["quality"]

ss_reg = ShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=12
)

for train_idx, test_idx in ss_reg.split(X_reg):
    X_train_reg = X_reg.iloc[train_idx]
    X_test_reg = X_reg.iloc[test_idx]
    y_train_reg = y_reg.iloc[train_idx]
    y_test_reg = y_reg.iloc[test_idx]

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(X_train_reg, y_train_reg)

train_pred_reg = linear_model.predict(X_train_reg)
test_pred_reg = linear_model.predict(X_test_reg)

reg_metrics = pd.DataFrame({
    "split": ["train", "test"],
    "MAE": [
        mean_absolute_error(y_train_reg, train_pred_reg),
        mean_absolute_error(y_test_reg, test_pred_reg)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_train_reg, train_pred_reg)),
        np.sqrt(mean_squared_error(y_test_reg, test_pred_reg))
    ],
    "R2": [
        r2_score(y_train_reg, train_pred_reg),
        r2_score(y_test_reg, test_pred_reg)
    ]
})

print("Regresja liniowa:\n")
print(reg_metrics)



print("\nWalidacja krzyżowa - Regresja liniowa")

cv_reg = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_reg = cross_validate(
    linear_model,
    X_reg,
    y_reg,
    cv=cv_reg,
    scoring={
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
)

print(f"Średnie MAE: {-cv_scores_reg['test_mae'].mean():.3f}")
print(f"Średnie R2 : {cv_scores_reg['test_r2'].mean():.3f}")




# Zad 2 - Zmiana wyników po usunięciu kilku cech

X_less = X_reg.drop(["volatile acidity", "alcohol"], axis=1)

cv_scores_less = cross_validate(
    linear_model,
    X_less,
    y_reg,
    cv=cv_reg,
    scoring={"r2": "r2"}
)

print(f"R2 po usunięciu cech: {cv_scores_less['test_r2'].mean():.3f}")

df["quality_binary"] = (df["quality"] >= 6).astype(int)
X_cls = df.drop(["quality", "quality_binary"], axis=1)
y_cls = df["quality_binary"]

ss_cls = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=12
)

for train_idx, test_idx in ss_cls.split(X_cls, y_cls):
    X_train_cls = X_cls.iloc[train_idx]
    X_test_cls = X_cls.iloc[test_idx]
    y_train_cls = y_cls.iloc[train_idx]
    y_test_cls = y_cls.iloc[test_idx]

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

log_model.fit(X_train_cls, y_train_cls)

train_pred_cls = log_model.predict(X_train_cls)
test_pred_cls = log_model.predict(X_test_cls)

train_prob = log_model.predict_proba(X_train_cls)[:, 1]
test_prob = log_model.predict_proba(X_test_cls)[:, 1]

print("\nTest:")
print(f"Accuracy : {accuracy_score(y_test_cls, test_pred_cls):.3f}")
print(f"Precision: {precision_score(y_test_cls, test_pred_cls):.3f}")
print(f"Recall   : {recall_score(y_test_cls, test_pred_cls):.3f}")
print(f"F1-score : {f1_score(y_test_cls, test_pred_cls):.3f}")
print(f"ROC AUC  : {roc_auc_score(y_test_cls, test_prob):.3f}")





# Zad 3 - Porównanie regresji logistycznej z DecisionTreeClassifier

print("\n\nWalidacja krzyżowa - Regresja logistyczna\n")
cv_cls = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=12
)

cv_scores_cls = cross_validate(
    log_model,
    X_cls,
    y_cls,
    cv=cv_cls,
    scoring={
        "acc": "accuracy",
        "f1": "f1",
        "roc": "roc_auc"
    }
)

print(f"Średnie Accuracy: {cv_scores_cls['test_acc'].mean():.3f}")
print(f"Średnie F1      : {cv_scores_cls['test_f1'].mean():.3f}")
print(f"Średnie ROC AUC : {cv_scores_cls['test_roc'].mean():.3f}")


print("\n\nPorównanie z DecisionTreeClassifier")

tree_model = DecisionTreeClassifier(random_state=12)

tree_scores = cross_validate(
    tree_model,
    X_cls,
    y_cls,
    cv=cv_cls,
    scoring={
        "acc": "accuracy",
        "f1": "f1"
    }
)

print("\nLogistic Regression:")
print(f"Accuracy: {cv_scores_cls['test_acc'].mean():.3f}")
print(f"F1:       {cv_scores_cls['test_f1'].mean():.3f}")

print("\nDecision Tree:")
print(f"Accuracy: {tree_scores['test_acc'].mean():.3f}")
print(f"F1:       {tree_scores['test_f1'].mean():.3f}")


# Macierz pomyłek


fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test_cls, test_pred_cls)).plot(ax=ax)
ax.set_title("Regresja logistyczna: macierz pomyłek")
plt.show()


sys.stdout.close()
