from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> list[Feature]:
    """Detects the feature types in a dataset.
    Assumption: only categorical and numerical features and no NaN values.

    Arguments:
        dataset (Dataset): Dataset containing features to read

    Returns:
        list[Feature]: List of features with their types.
    """
    data = dataset.read_as_data_frame()
    features = []

    for column in data.columns:
        if data[column].dtype.kind in 'biufc':
            feature = Feature(name=column, type='numerical')
        else:
            feature = Feature(name=column, type='categorical')
        features.append(feature)
    return features
