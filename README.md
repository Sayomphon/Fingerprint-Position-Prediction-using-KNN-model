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
**df**

![Result](https://github.com/Sayomphon/Fingerprint-Position-Prediction-using-KNN-model/blob/main/df.PNG)

## Section 4: Handle Missing Values
In this segment, the code checks for any missing values in the dataset using *df.isnull().sum()*.
- It then fills those missing values with the mean of their respective columns to ensure that the model can be trained without issues caused by missing data.
```python
# Check for missing values
print(df.isnull().sum())
```
![Result](https://github.com/Sayomphon/Fingerprint-Position-Prediction-using-KNN-model/blob/main/dfisnull.PNG)

```python
# Replace missing values (example: using mean)
df.fillna(df.mean(), inplace=True)
```
## Section 5: Separate Features and Target Variable
Here, the code separates the features (independent variables) from the target variable (dependent variable).
- *X* contains all the columns except the target, which is denoted by 'TARGET', and *y* contains the 'TARGET' column.
```python
X = df.drop('TARGET', axis=1)
y = df['TARGET']
```
## Section 6: Split the Data
The dataset is split into training and testing sets using an 80/20 split.
- *train_test_split()* from sklearn randomly divides the data while allowing for reproducibility using a *random_state*.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Section 7: Create KNN Model
A KNN classifier is instantiated with *n_neighbors=3*, and the model is trained using the training dataset.
```python
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
```
## Section 8: Make Predictions
Predictions are made on the test dataset using the trained KNN model.
```python
y_pred = model.predict(X_test)
```
**KNN model**

![Result](https://github.com/Sayomphon/Fingerprint-Position-Prediction-using-KNN-model/blob/main/KNN%20model.PNG)

## Section 9: Evaluate the Model
This section evaluates the model's performance:
- It calculates and prints the accuracy score.
- It generates a classification report showing precision, recall, and f1-score.
- A confusion matrix is created and visualized using seaborn to understand the model’s predictions versus actual outcomes.
```python
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

**accuracy**

![Result](https://github.com/Sayomphon/Fingerprint-Position-Prediction-using-KNN-model/blob/main/accuracy.PNG)

```python
# Display Classification Report
print(classification_report(y_test, y_pred))
```
**classification report**

![Result](https://github.com/Sayomphon/Fingerprint-Position-Prediction-using-KNN-model/blob/main/classification%20report.PNG)
