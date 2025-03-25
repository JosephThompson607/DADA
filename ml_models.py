from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train, X_test, y_test, random_state=42, n_estimators=100, max_depth=5, class_weight='balanced'):
    # Initialize the model
    rf_model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, class_weight=class_weight)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_model.predict(X_test)

    # Evaluate performance
    print(classification_report(y_test, y_pred))
    return rf_model

def train_log_regression(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('model', LogisticRegression(class_weight='balanced', solver='liblinear'))  # Use liblinear solver for medium datasets
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return pipeline




def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=5):
    # Initialize the model
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, class_weight='balanced')
    #dt_model = DecisionTreeClassifier(random_state=42, max_depth=7, )

    # Train the model
    dt_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = dt_model.predict(X_test)

    # Evaluate performance
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    return dt_model

def plot_decision_tree(X_train, dt_model):
    plt.figure(figsize=(30, 20))  # Adjust size for better readability
    plot_tree(dt_model, filled=True, feature_names=X_train.columns, class_names=["Class 0", "Class 1"])
    plt.savefig('decision_tree.png')


def random_forest_grid_search(X_train, y_train, X_test, y_test):
    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],      # More trees improve stability
        "max_depth": [5, 10, 20, None],       # Control tree depth
        "min_samples_split": [2, 5, 10],      # Allow splitting on fewer samples
        "min_samples_leaf": [1, 3, 5],        # Smaller leaves detect minority class
        "class_weight": ["balanced", "balanced_subsample"],  # Handles imbalance
    }

    # Grid search with recall optimization
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="recall",  # Optimize for recall
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    # Best parameters
    print("Best parameters:", grid_search.best_params_)

    # Train the best model
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)

    print(classification_report(y_test, y_pred))
    return best_rf_model

def plot_feature_importances(rf_model, X_train, title = "Feature Importance in Random Forest"):
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 5))
    feature_importances[:10].plot(kind="bar")  # Show top 10 features
    plt.title(title)
    plt.show()
    return feature_importances