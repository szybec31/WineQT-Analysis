from sklearn.model_selection import cross_validate

def evaluate_model(model, X, y, cv, scoring):

    scores = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    return {
        "f1": f"{scores['test_f1'].mean():.3f} ({scores['test_f1'].std():.3f})",
        "bal_acc": f"{scores['test_bal_acc'].mean():.3f} ({scores['test_bal_acc'].std():.3f})",
        "precision": f"{scores['test_precision'].mean():.3f} ({scores['test_precision'].std():.3f})",
        "recall": f"{scores['test_recall'].mean():.3f} ({scores['test_recall'].std():.3f})",
    }