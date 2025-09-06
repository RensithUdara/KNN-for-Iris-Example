# Iris Flower Classification using KNN

This repository contains a machine learning project that demonstrates how to build, train, and use a K-Nearest Neighbors (KNN) classifier to identify different types of Iris flowers based on their measurements.

## Project Overview

The Iris dataset is a classic dataset in machine learning and statistics. It contains measurements of four features (sepal length, sepal width, petal length, and petal width) for 150 flowers from three different Iris species. This project uses the KNN algorithm to create a model that can predict the species of an Iris flower based on these four measurements.

## Repository Contents

- `1-KNN-for-Iris-example.ipynb`: Jupyter notebook containing the full model development process, including:
  - Data loading and inspection
  - Data preprocessing
  - Model training
  - Model evaluation
  - Model saving
  
- `getting-prediction-KNN.ipynb`: Jupyter notebook demonstrating how to load and use the trained model for making new predictions

- `iris.csv`: The dataset used for training and testing the model

- `knn_model.pkl` and `knn_model.sav`: Saved model files that can be loaded to make predictions without retraining

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib
- Jupyter Notebook/Lab

## Usage Instructions

### Training the Model

1. Open the `1-KNN-for-Iris-example.ipynb` notebook in Jupyter Notebook or Jupyter Lab.
2. Run all cells in sequence to:
   - Load and explore the Iris dataset
   - Split the data into training and testing sets
   - Train a KNN classifier model
   - Evaluate model accuracy
   - Save the model to disk

### Making Predictions with the Trained Model

1. Open the `getting-prediction-KNN.ipynb` notebook in Jupyter Notebook or Jupyter Lab.
2. Run all cells to:
   - Load the pre-trained model
   - Create test data (flower measurements)
   - Generate predictions about the Iris species

### Sample Prediction

The prediction notebook includes a sample of how to use the model to classify new Iris flowers:

```python
import joblib
model = joblib.load("knn_model.sav")

# Sample measurements: [sepal_length, sepal_width, petal_length, petal_width]
test = [[5.1, 3.5, 1.4, 0.2],  # Sample 1
        [6.7, 3.1, 4.7, 1.5],  # Sample 2
        [7.2, 3.6, 6.1, 2.5]]  # Sample 3

result = model.predict(test)
print(result)  # Output: class predictions (0: Setosa, 1: Versicolor, 2: Virginica)
```

## Model Information

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (Iris Setosa, Iris Versicolor, Iris Virginica)
- **Performance**: The model achieves high accuracy on the test set (typically >90%)

## About the Iris Dataset

The Iris dataset contains 150 samples of Iris flowers from three species:
- Class 0: Iris Setosa
- Class 1: Iris Versicolor
- Class 2: Iris Virginica

Each sample has four features measured in centimeters:
- Sepal length
- Sepal width
- Petal length
- Petal width

## License

This project is open source and available for educational purposes.

## Author

RensithUdara

## Acknowledgments

- The Iris dataset is a classic in the field of machine learning
- Special thanks to the scikit-learn team for providing easy-to-use machine learning tools
