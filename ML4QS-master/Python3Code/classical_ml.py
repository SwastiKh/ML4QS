import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
import random

import sklearn.metrics



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
    model = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=23)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    
    return accuracy


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
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    
    return accuracy
