import unittest
import pickle
import pandas as pd
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.classification import KNearestNeighborsClassification
from autoop.core.ml.metric import AccuracyMetric, RecallMetric, F1Score, PrecisionMetric

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


class TestModels(unittest.TestCase):

    def setUp(self) -> None:
        df = pd.read_csv("assets/objects/iris_data.csv")
        self.dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=KNearestNeighborsClassification(),
            input_features=list(filter(lambda x: x.name != "Name", self.features)),
            target_feature=Feature(name="Name", type="categorical"),
            metrics=[AccuracyMetric(), PrecisionMetric(), RecallMetric(), F1Score()],
            split=0.8
        )

    def test_init(self):
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()

    def test_train(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        knn_model = KNeighborsClassifier(n_neighbors=3)
        x_train = self.pipeline._compact_vectors(vectors=self.pipeline._train_X)
        knn_model.fit(x_train, self.pipeline._train_y)
        x_test = self.pipeline._compact_vectors(vectors=self.pipeline._test_X)
        self.assertEqual(self.pipeline._predictions.all(), knn_model.predict(x_test).all())
        #self.assertEqual(self.pipeline._metrics_results[0][1], accuracy_score(self.pipeline._predictions, self.pipeline._test_y))
        self.assertEqual(self.pipeline._metrics_results[1][1], precision_score(self.pipeline._predictions, self.pipeline._test_y, average='micro'))
        self.assertEqual(self.pipeline._metrics_results[2][1], recall_score(self.pipeline._predictions, self.pipeline._test_y, average='micro'))
        self.assertEqual(self.pipeline._metrics_results[3][1], f1_score(self.pipeline._predictions, self.pipeline._test_y, average='micro'))

    def test_artifacts(self):
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline.artifacts)
        model_artifact = self.pipeline.artifacts[-1].read()
        model = pickle.loads(model_artifact)
