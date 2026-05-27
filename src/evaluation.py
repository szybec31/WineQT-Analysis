from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X, y, cv, scoring):

    scores = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    return {
        "f1_macro": f"{scores['test_f1_macro'].mean():.3f} ({scores['test_f1_macro'].std():.3f})",
        "bal_acc": f"{scores['test_bal_acc'].mean():.3f} ({scores['test_bal_acc'].std():.3f})",
        "precision": f"{scores['test_precision'].mean():.3f} ({scores['test_precision'].std():.3f})",
        "recall": f"{scores['test_recall'].mean():.3f} ({scores['test_recall'].std():.3f})",
    }

def mean_confusion_matrix(model, X, y, cv):
    cms = []
    labels = np.unique(y)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        cms.append(cm)

    mean_cm = np.mean(cms, axis=0)

    return mean_cm, labels

def plot_roc_curves(models, X, y, cv):
    plt.figure(figsize=(8, 6))  # W przeciwnym wypadku za duża legenda imo

    for name, model in models:
        y_true = []
        y_prob_all = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_true.extend(y_test)
            y_prob_all.extend(y_prob)

        fpr, tpr, _ = roc_curve(y_true, y_prob_all, pos_label=6)
        auc_score = roc_auc_score(y_true, y_prob_all)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

