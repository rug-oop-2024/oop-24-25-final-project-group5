
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """

    features = []
    read_data = dataset.read()

    for column in read_data.columns:
        # Go through items inside the column, if they are all numerical, then it is numerical feature.
        if read_data[column].apply(lambda x: isinstance(x, (int, float))).all():
            features.append(Feature(column, "numerical"))
        else:
            features.append(Feature(column, "categorical"))

    return features