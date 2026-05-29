from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_rf_resampling_confusion_matrices(
        X,
        y,
        cv,
        models_dict,
        resamplers_dict,
        create_pipeline
):
    """
    Wyświetla średnie macierze pomyłek dla:
    - RandomForest + none
    - RandomForest + SMOTE
    - RandomForest + ROS
    """

    rf_model = models_dict["RandomForestClassifier"]

    selected_resamplers = ["none", "smote", "ros"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = np.unique(y)

    for idx, res_name in enumerate(selected_resamplers):

        resampler = resamplers_dict[res_name]

        # RF nie potrzebuje skalowania
        pipe = create_pipeline(
            rf_model,
            resampler=resampler,
            use_scaler=False
        )

        mean_cm, labels = mean_confusion_matrix(
            pipe,
            X,
            y,
            cv
        )

        sns.heatmap(
            mean_cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx]
        )

        axes[idx].set_title(f"RandomForest + {res_name.upper()}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")

    plt.suptitle(
        "RandomForest - Mean Confusion Matrices",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        "charts/random_forest_resampling_confusion_matrices.png"
    )

    plt.show()

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
    plt.savefig("charts/roc_curve.png")
    plt.show()

