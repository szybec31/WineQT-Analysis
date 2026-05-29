from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(model, resampler=None, use_scaler=True):

    steps = []

    # Skalowanie (ważne np. dla LogisticRegression i SMOTE)
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    # Resampling (SMOTE / ROS)
    if resampler is not None:
        steps.append(("resample", resampler))

    # Model
    steps.append(("model", model))

    return Pipeline(steps)