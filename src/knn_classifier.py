import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from typing import Optional

class KNNClassifier:
    def __init__(self, k: int = 3) -> None:
        """
        K-Nearest Neighbors classifier.

        :param k: Number of nearest neighbors to consider for voting.
        """
        self.k = k
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the classifier to the training data.

        :param X: Training features.
        :param y: Training labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided test data.

        :param X: Test features.
        :return: Predicted labels for each test sample.
        """
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x: np.ndarray) -> int:
        """
        Predict the label for a single test sample.

        :param x: A single test sample.
        :return: Predicted label.
        """
        # Compute the L2 distances between x and all training samples
        distances = np.linalg.norm(self.X_train - x, axis=1)
        
        # Get the indices of the k-nearest neighbors
        k_nearest_indices = np.argpartition(distances, self.k)[:self.k]

        # Get the most common target among the nearest neighbors
        k_nearest_targets = self.y_train[k_nearest_indices]
        most_common_target = Counter(k_nearest_targets).most_common(1)

        return most_common_target[0][0]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of the classifier.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Accuracy as a float.
    """
    return np.sum(y_true == y_pred) / len(y_true)


def main() -> None:
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the KNN classifier
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)

    # Predict the labels for the test set
    predictions = knn.predict(X_test)

    # Print results
    print("Predictions:", predictions)
    print("True labels:", y_test)
    
    # Compute accuracy
    acc = accuracy(y_test, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")

if __name__ == '__main__':
    main()
