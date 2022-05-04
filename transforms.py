import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
import utils


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Custom class to encode binary features.
    This class inherits from sklearn.base.BaseEstimator and
    sklearn.base.TransformerMixin. Hence it works with all sklearn pipelines and
    ColumnTransformers
    """
    def __init__(self, **kwargs):
        super(BinaryEncoder).__init__(**kwargs)
        self.fitted = False
        self.keys = []
        self.categories_ = []

    def fit(self, X, y=None):
        self.fitted = True
        for j in range(X.shape[1]):
            # Get unique values in that column
            bin_vals = np.unique(X[:, j])
            self.categories_.append(bin_vals)
            # print(bin_vals)
            self.keys.append({val: i for i, val in enumerate(bin_vals)})
        return self

    def transform(self, X, y=None):
        if not self.fitted:
            raise Exception("This instance of BinaryEncoder is not fitter. "
                            "Call fit() before predict")
        else:
            output = np.zeros_like(X)
            for j in range(X.shape[1]):
                output[:, j] = np.array(list(map(self.keys[j].get, X[:, j])))
            return output


class RemoveFeatures(BaseEstimator, TransformerMixin):
    """
    Custom class to remove features.
    This class inherits from sklearn.base.BaseEstimator and
    sklearn.base.TransformerMixin. Hence, it works with all sklearn pipelines
    and ColumnTransformers
    """
    def __init__(self):
        super(RemoveFeatures).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.delete(X, range(X.shape[1]), axis=1)
        X = X.astype(int)
        return X


class IdentityTransform(BaseEstimator, TransformerMixin):
    """
    Custom class to retain features as is.
    This class inherits from sklearn.base.BaseEstimator and
    sklearn.base.TransformerMixin. Hence, it works with all sklearn pipelines
    and ColumnTransformers
    """
    def __init__(self):
        super(IdentityTransform).__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


if __name__ == "__main__":
    datapoints, labels, feature_info, categories = utils.read_data(
        "datasets/student_performance_train.csv",
        "datasets/student_performance_feature.txt")
    test_data, test_labels, _, _ = utils.read_data(
        "datasets/student_performance_test.csv",
        "datasets/student_performance_feature.txt")

    ct = ColumnTransformer([
        ("BinaryEncoder", BinaryEncoder(),
         np.where(feature_info == "binary")[0]),
        ("Identity",
         IdentityTransform(),
         np.where(feature_info == "numeric")[0]),
        ("RemoveNominal", RemoveFeatures(),
         np.where(feature_info == "nominal")[0]),
    ])

    ct.fit(datapoints)
    a = ct.transform(datapoints)
    print(a.dtype)
    print(a.shape)