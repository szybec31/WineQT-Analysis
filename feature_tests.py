from src.data_loader import load_data
from src.preprocessing import prepare_data

from src.feature_statistical_tests import (
    compare_feature_between_classes
)

# =========================================
# LOAD
# =========================================

df = load_data("Dataset/WineQT.csv")

# =========================================
# BINARY MODE
# =========================================

_, y = prepare_data(df, mode="binary")

df = df.drop(["Id"], axis=1)

df["quality"] = y

# =========================================
# TESTED FEATURES
# =========================================

features = [
    "alcohol",
    "sulphates",
    "volatile acidity"
]

# =========================================
# TESTS
# =========================================

for feature in features:

    compare_feature_between_classes(
        df,
        feature=feature,
        class_a=5,
        class_b=6
    )
