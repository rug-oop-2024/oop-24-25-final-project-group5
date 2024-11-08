
import unittest
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline
from autoop.tests.test_models import TestModels
from autoop.tests.test_decision_tree_classification import TestDecisionTreeClassification
from autoop.tests.test_k_nearest_neighbors_classification import TestKNearestNeighborsClassification
from autoop.tests.test_random_forest_classification import TestRandomForestClassification
from autoop.tests.test_lasso_regression import TestLassoRegression
from autoop.tests.test_multiple_linear_regression import TestMultipleLinearRegression
from autoop.tests.test_polynomial_regression import TestPolynomialRegression

if __name__ == '__main__':
    unittest.main()