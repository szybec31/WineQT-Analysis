from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

from sklearn.metrics import make_scorer, precision_score

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=False, allow_writing_files=False, random_state=42),
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
    #"smoteenn": SMOTEENN(random_state=42)
}

PARAM_GRIDS = {

    "GradientBoostingClassifier": {
        # Demo z 5 x 4 x 3 = 60 kombinacji
        "model__n_estimators": range(100, 301, 50),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "model__max_depth": [3, 4, 5],


        # Do ostatecznego testu 9 x 6 x 4 = 216 kombinacji
        # "model__n_estimators": range(100, 301, 25),
        # "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        # "model__max_depth": [3, 4, 5, 6]

    },

    "CatBoost": {
        # Demo z 5 x 4 x 3 = 60 kombinacji
        "model__iterations": range(100, 301, 50),
        "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "model__depth": [3, 4, 5]

        # Do ostatecznego testu 9 x 6 x 4 = 216 kombinacji
        # "model__iterations": range(100, 301, 25),
        # "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        # "model__depth": [3, 4, 5, 6]
    }
}

