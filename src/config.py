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
    "RandomForestClassifier": RandomForestClassifier(
        #max_depth=None,
        #max_features="sqrt",
        #min_samples_leaf=2,
        #min_samples_split=5,
        #n_estimators=300,
        random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(
        #depth=8,
        #iterations=500,
        #l2_leaf_reg=3,
        #learning_rate=0.1,
        random_state=42,allow_writing_files=False,verbose=False)
}

SCORING = {
    "f1_macro": "f1_macro",
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

    "RandomForestClassifier": {
        # 3 × 3 × 3 × 3 × 2 = 162 kombinacje
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", "log2"]

        # Do ostatecznego testu 9 x 6 x 4 = 216 kombinacji
        # "model__n_estimators": range(100, 301, 25),
        # "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        # "model__max_depth": [3, 4, 5, 6]

    },

    "CatBoost":
    {
        # 2 × 2 × 3 × 3 = 36 kombinacji
        "model__iterations": [200, 500],
        "model__learning_rate": [0.03, 0.1],
        "model__depth": [4, 6, 8],
        "model__l2_leaf_reg": [3, 5, 7]

        # Do ostatecznego testu 9 x 6 x 4 = 216 kombinacji
        # "model__iterations": range(100, 301, 25),
        # "model__learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
        # "model__depth": [3, 4, 5, 6]
    }
}

