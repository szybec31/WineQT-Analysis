from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

from sklearn.metrics import make_scorer, precision_score

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
}

SCORING = {
    "f1": "f1_macro",
    "bal_acc": "balanced_accuracy",
    "precision": make_scorer(precision_score, average="macro", zero_division=0),
    "recall": "recall_macro"
}

RESAMPLERS = {
    "none": None,
    "smote": SMOTE(k_neighbors=1,random_state=42),
    "ros": RandomOverSampler(random_state=42),
    "smoteenn": SMOTEENN(random_state=42)
}