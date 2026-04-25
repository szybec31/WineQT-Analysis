from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def create_pipeline(model, use_smote=True, use_scaler=True):
    steps = []

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if use_smote:
        steps.append(("smote", SMOTE(k_neighbors=2, random_state=42)))

    steps.append(("model", model))

    return Pipeline(steps)