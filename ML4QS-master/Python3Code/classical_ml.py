import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def plot_correlation_matrix(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

df = pd.read_csv("all_features.csv", engine='python')


# convert 'exercise' to categorical
df['exercise'] = df['exercise'].astype('category')
# convert 'exercise' to numerical codes
df['exercise_code'] = df['exercise'].cat.codes
# check the unique codes
print("Unique exercise codes:", df['exercise_code'].unique())
# Unique exercises: ['burpees' 'crunches' 'jumping_jacks' 'plank' 'squats']



# Define features and labels
columns_to_drop=['exercise', 'participant' , 'dataset' , 'exercise_participant', 'gyro_x_mean', 'gyro_x_std', 'gyro_x_min', 'gyro_x_max', 'gyro_x_range',
       'gyro_x_median', 'gyro_x_q25', 'gyro_x_q75', 'gyro_x_iqr',
       'gyro_x_energy', 'gyro_x_rms', 'gyro_magnitude_mean',
       'gyro_magnitude_std', 'gyro_magnitude_max', 'exercise_code']
existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
X = df.drop(columns=existing_cols_to_drop, errors='ignore')

y = df['exercise_code']

plot_correlation_matrix(X)


# Train-test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)


# Ramdom forest classification for classifyung exercises
def random_forest_classification(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10):
    """
    Train a Random Forest classifier and evaluate its performance.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Test feature set
    - y_test: Test labels
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth of the trees
    
    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    
    # Create and fit the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=23)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    
    return accuracy, model


def knn_classification(X_train, y_train, X_test, y_test, n_neighbors=5):
    """
    Train a K-Nearest Neighbors classifier and evaluate its performance.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Test feature set
    - y_test: Test labels
    - n_neighbors: Number of neighbors to use
    
    Returns:
    - accuracy: Accuracy of the model on the test set
    """
    
    # Create and fit the model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    
    return accuracy

def linear_regression(X_train, y_train, X_test, y_test):
    """
    Train a Linear Regression model and evaluate its performance.
    
    Parameters:
    - X_train: Training feature set
    - y_train: Training labels
    - X_test: Test feature set
    - y_test: Test labels
    
    Returns:
    - r2_score: R^2 score of the model on the test set
    """
    
    from sklearn.linear_model import LinearRegression
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate R^2 score
    r2_score = sklearn.metrics.r2_score(y_test, predictions)
    
    return r2_score

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:][::-1]  # Top N features

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# Run models and print results
rf_acc, rf_model = random_forest_classification(X_train, y_train, X_test, y_test)
knn_acc = knn_classification(X_train, y_train, X_test, y_test)
linear_r2 = linear_regression(X_train, y_train, X_test, y_test)
plot_feature_importance(rf_model, X.columns)


print(f'Random Forest Accuracy: {rf_acc:.4f}')
print(f'KNN Accuracy: {knn_acc:.4f}')
print(f'Linear Regression R^2 Score: {linear_r2:.4f}')