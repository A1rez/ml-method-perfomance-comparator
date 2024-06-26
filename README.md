# Machine Learning Model Comparison on Iris Dataset

This project evaluates the performance of four different machine learning models on the Iris dataset: K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), and Multi-layer Perceptron (MLP). The script trains each model, evaluates their performance, and visualizes the results.

## Getting Started

### Prerequisites

To run this project, you need to have Python installed. You can install the necessary Python packages using the provided `requirements.txt` file.

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/A1rez/ml-method-perfomance-comparator
    ```

2. Change into the project directory:
    ```sh
    cd ml-method-perfomance-comparator
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the script to train and evaluate the models. The script will:
1. Load the Iris dataset.
2. Split the dataset into training and test sets.
3. Train and evaluate the KNN, Random Forest, SVM, and MLP models.
4. Plot the performance metrics of all models.
5. Optionally, display confusion matrices for any chosen model.

To run the script, use:
```sh
python code.py

## Script Details

KNNeighbors: Trains and evaluates a K-Nearest Neighbors model.
RandomForest: Trains and evaluates a Random Forest model.
SvM: Trains and evaluates a Support Vector Machine model.
MlP: Trains and evaluates a Multi-layer Perceptron model.
plot_performance: Plots the performance metrics of all models using a heatmap.
confu_plot: Plots confusion matrices for training and test data of a selected model.

## Example Output

The script will display a heatmap showing the performance metrics (accuracy, precision, recall, f1-score) of all models. If you choose to display confusion matrices, they will be plotted for the chosen model.

## Acknowledgments

The script uses the Iris dataset from the UCI Machine Learning Repository.
The models are implemented using scikit-learn.
