from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def create_pipeline(model, resampler=None, use_scaler=True):
    steps = []

    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    if resampler is not None:
        steps.append(("resample", resampler))

    steps.append(("model", model))

    return Pipeline(steps)