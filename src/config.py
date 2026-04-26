from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
}

SCORING = {
    "f1": "f1_macro",
    "bal_acc": "balanced_accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro"
}