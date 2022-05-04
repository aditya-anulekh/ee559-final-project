import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.model_selection import cross_validate


def read_data(filepath, feature_path):
    """
    Function to read data from a CSV file and return the data and labels. i and
    j are the feature numbers.
    :param filepath: str
    :param feature_path: str
    :return: tuple(np.array, np.array)
    """
    data = pd.read_csv(filepath)  # Read the CSV file using pandas
    datapoints = np.array(data.iloc[:, 0:-3])  # Array to store the coordinates
    labels = np.array(data.iloc[:, -3:])  # Array to store the labels

    # List to store the type of each feature
    feature_info = []
    all_categories = []
    with open(feature_path, "r") as file:
        txt = file.readlines()
        for line in txt:
            feature_type = re.findall("\(\w+:", line)[0][1:-1]
            feature_info.append(feature_type)
            if feature_type == "nominal":
                categories = re.findall("'\w*'", line)
                categories = [s.replace("'", "") for s in categories]
                categories = np.array(categories)
                all_categories.append(categories)
            else:
                all_categories.append(np.array([0]))
    return datapoints, labels, np.array(feature_info), all_categories


def report_metrics(y_actual, y_pred, **kwargs):
    fig, _ = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    fig.tight_layout()
    rmse = mean_squared_error(y_actual, y_pred, squared=False)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    fig.axes[0].hist(y_actual)
    fig.axes[0].set_title("Actual Distribution")
    fig.axes[1].hist(y_pred)
    fig.axes[1].set_title("Predicted Distribution")
    print(f"Root Mean Square Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-Squared score: {r2}")
    return rmse, mae, r2, fig


def mutual_info_regression_seeded(X, y, random_state=43):
    """
    Seeded function to reproduce results from mutual_information
    """
    return mutual_info_regression(X, y, random_state=random_state)


def euclidean_distance(point1, point2):
    """
    Function to calculate euclidean distance between two points
    """
    assert point1.shape == point2.shape, f"Input points have different shapes " \
                                         f"{point1.shape}, {point2.shape}"
    return np.sqrt(np.sum((point1 - point2) ** 2))


def get_information_plots(X, y):
    """
    Function to plot each feature with the output feature
    The title of each plot represents the f_regression score and
    mutual_information_score as a ratio with the max score
    """
    num_features = X.shape[1]
    fig, _ = plt.subplots(num_features // 9 + 1, 9)
    fig.set_size_inches(18, (num_features // 9 + 1) * 2)
    fig.tight_layout()

    f_test, _ = f_regression(X, y)
    f_test /= np.max(f_test)

    mi = mutual_info_regression(X, y, random_state=43)
    mi /= np.max(mi)

    for i in range(X.shape[1]):
        fig.axes[i].scatter(X[:, i], y)
        fig.axes[i].set_title(f"F:{f_test[i]:0.2f}, M:{mi[i]:0.2f}")
    return fig


class ModelSelector:
    """
    Wrapper around sklearn.model_selection.cross_validate to perform
    cross validation on a given pipeline and return the best estimator
    """
    def __init__(self, pipeline):
        self.fitted = False
        self.pipeline = pipeline
        self.best_estimator = None

    def fit(self, X, y, search_space):
        self.fitted = True
        scores = {'test_score': [],
                  'train_score': []}
        # List to store the estimator outputs from each cross validation
        outputs = []
        fig = plt.figure()
        for k in tqdm(search_space):
            score = cross_validate(self.pipeline(k), X,
                                   y, return_train_score=True,
                                   return_estimator=True)
            scores['test_score'].append(score['test_score'].mean())
            scores['train_score'].append(score['train_score'].mean())
            outputs.append(score)

        plt.plot(search_space, scores['train_score'], label="train")
        plt.plot(search_space, scores['test_score'], label="test")
        plt.plot(search_space[np.argmax(scores["test_score"])],
                 max(scores["test_score"]), "x", markersize=5,
                 label="optimal value")
        plt.legend()

        best_cv_output = outputs[np.argmax(scores["test_score"])]
        self.best_estimator = best_cv_output["estimator"][np.argmax(
            best_cv_output["test_score"]
        )]
        return best_cv_output, search_space[np.argmax(scores["test_score"])], \
               fig

    def get_best_estimator(self):
        if not self.fitted:
            raise Exception("This instance of ModelSelector is not fitted. "
                            "Call fit() before calling predict")
        return self.best_estimator
