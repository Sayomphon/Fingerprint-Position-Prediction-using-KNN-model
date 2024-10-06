# Fingerprint Position Prediction using KNN-model

The code imports necessary libraries for data manipulation, modeling, and visualization. It loads a dataset, checks for missing values, and replaces them with the mean. The data is then split into training and testing sets, and a K-Nearest Neighbors (KNN) model is created and trained. Predictions are made on the test set, and the model's performance is evaluated using accuracy, a classification report, and a confusion matrix. Lastly, it tests different values of K to find the optimal number of neighbors for the KNN algorithm.

## Section 1: Import Necessary Libraries
This section imports the required libraries to perform data analysis and machine learning tasks:
- **pandas** and **numpy** for data manipulation and numerical operations.
- **sklearn** for model building, specifically using **K-Nearest Neighbors (KNN) classification**.
- **matplotlib** and **seaborn** for data visualization.
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```
## Section 2: Load Dataset
Here, the code loads a dataset named *FP_DATASET_Fullcut2.csv* into a DataFrame using pandas.
```python
df = pd.read_csv('FP_DATASET_Fullcut2.csv')
```
## Section 3: Check the Initial Data Overview
This part provides an overview of the dataset:
- *The df.info()* command displays the structure, including data types and non-null counts.
- *The df.describe()* command provides statistical summaries of numerical features.
```python
print(df.info())
print(df.describe())
```
![Result](https://github.com/Sayomphon/Weather-forecast-model/blob/main/Prediction%20result.PNG)

![Result](https://github.com/Sayomphon/Fingerprint-Position-Prediction-using-KNN-model/blob/main/df.PNG)
