
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
    data = dataset.read()
    features = []

    for column in data.columns:
        if data[column].dtype.kind in 'biufc':
            feature = Feature(name=column, type='numerical')
        else:
            feature = Feature(name=column, type='categorical')
        features.append(feature)
    return features
