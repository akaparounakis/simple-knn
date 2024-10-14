# simple-knn

A lightweight implementation of the K-Nearest Neighbors (KNN) algorithm from scratch using NumPy. Ideal for educational purposes and small-scale machine learning projects.

## Features

- Custom KNN classifier without external dependencies like `scikit-learn` (except for data loading and splitting).
- Works with any dataset that fits into memory, as long as it is in NumPy array format.
- Easy to read and modify, making it great for learning and experimentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simple-knn.git
   cd simple-knn
   ```

2. Install the required dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage / Examples

### Training and Testing the Model
You can run the `main()` function to train and test the KNN classifier on the Iris dataset:

```bash
python knn_classifier.py
```

The output will show predictions and the true labels for the test set, along with the accuracy of the model.

### Example Output
```bash
Predictions: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
True labels: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]
Accuracy: 100.00%
```

### Modify the `k` Value
You can easily modify the number of neighbors by changing the `k` value in the `KNNClassifier` instantiation:

```python
knn = KNNClassifier(k=5)
```

## How It Works
1. The classifier calculates the Euclidean distance between a test sample and all training samples.
2. It finds the `k` nearest neighbors based on this distance.
3. The majority class among the neighbors is assigned as the predicted label.
