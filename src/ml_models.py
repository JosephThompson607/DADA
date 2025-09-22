from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import json

def train_random_forest(X_train, y_train, X_test, y_test, random_state=42, n_estimators=100, max_depth=5, class_weight='balanced', min_samples_split=2, min_samples_leaf=1, criterion='log_loss'):
    # Initialize the model
    rf_model = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth,criterion=criterion, class_weight=class_weight, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)


    # Evaluate performance
    print(classification_report(y_test, y_pred))
    return rf_model, (fpr, tpr, roc_auc)

def train_log_regression(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardize features
        ('model', LogisticRegression(class_weight='balanced', solver='liblinear'))  # Use liblinear solver for medium datasets
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve
    # Plot ROC curve

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print("ROC AUC:", roc_auc)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #return trained model and roc curve
    return pipeline, (fpr, tpr, roc_auc) 




def train_xg_boost(X_train, y_train, X_test, y_test,random_state=42, n_estimators=100,gamma=0, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0.01, min_child_weight=1):
    boost_graph = xgb.XGBClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=random_state,
    reg_lambda=reg_lambda, 
    reg_alpha=reg_alpha, 
    min_child_weight=min_child_weight,
    n_jobs=-1, 
        gamma=gamma
    )
    boost_graph.fit(X_train, y_train)

    # Make predictions
    y_pred_2 = boost_graph.predict(X_test)
    y_prob = boost_graph.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred_2)
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, y_pred_2))
    return boost_graph, (fpr,tpr, roc_auc)

def train_decision_tree(X_train, y_train, X_test, y_test, max_depth=5, random_state=42, criterion='log_loss', class_weight='balanced', min_samples_split=2, min_samples_leaf=1):
    # Initialize the model

    dt_model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, class_weight=class_weight, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
    #dt_model = DecisionTreeClassifier(random_state=42, max_depth=7, )

    # Train the model
    dt_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = dt_model.predict(X_test)
    y_prob = dt_model.predict_proba(X_test)[:, 1]  # Get probabilities for ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Evaluate performance
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    return dt_model, (fpr, tpr, roc_auc)

def plot_decision_tree(X_train, dt_model):
    plt.figure(figsize=(30, 20))  # Adjust size for better readability
    plot_tree(dt_model, filled=True, feature_names=X_train.columns, class_names=["Class 0", "Class 1"])
    plt.savefig('decision_tree.png')

def decision_tree_grid_search(X_train, y_train, X_test, y_test, 
                             res_name="decision_tree_grid_search.json", 
                             subsample=None, rand_search=True):
    """
    Perform grid search for Decision Tree with option for subsampling
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data  
    res_name : str, filename for saving results
    subsample : float, fraction of training data to use (e.g., 0.3 for 30%)
    rand_search : bool, whether to use RandomizedSearchCV vs GridSearchCV
    
    Returns:
    --------
    best_dt_model : trained model
    best_params : dict of best parameters
    """
    
    # Handle subsampling
    if subsample:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, 
            train_size=subsample,
            random_state=42, 
            stratify=y_train
        )
    else:
        X_sample = X_train.copy()
        y_sample = y_train.copy()
    
    # Define parameter grid for Decision Tree
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_features": [None, "sqrt", "log2"],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.01, 0.1]  # Pruning parameter
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    
    if rand_search:
        grid_search = RandomizedSearchCV(
            dt, param_grid,
            scoring='roc_auc',
            n_iter=20,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
    else:
        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
    
    # Fit the grid search
    grid_search.fit(X_sample, y_sample)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    # Get best model and make predictions
    best_dt_model = grid_search.best_estimator_
    y_pred = best_dt_model.predict(X_test)
    
    # Prepare results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'all_results': []
    }
    
    # Add all parameter combinations and their scores
    for i, params in enumerate(grid_search.cv_results_['params']):
        results['all_results'].append({
            'params': params,
            'mean_score': grid_search.cv_results_['mean_test_score'][i],
            'std_score': grid_search.cv_results_['std_test_score'][i]
        })
    
    # Save to JSON file
    with open(res_name, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {res_name}")
    print(classification_report(y_test, y_pred))
    
    return best_dt_model, grid_search.best_params_


def random_forest_grid_search(X_train, y_train, X_test, y_test, 
                             res_name="random_forest_tree_grid_search.json", 
                             subsample=None, rand_search=True):
    """
    Perform grid search for Random Forest with option for subsampling
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data  
    res_name : str, filename for saving results
    subsample : float, fraction of training data to use (e.g., 0.3 for 30%)
    rand_search : bool, whether to use RandomizedSearchCV vs GridSearchCV
    
    Returns:
    --------
    best_rf_model : trained model
    best_params : dict of best parameters
    """

    if subsample:
        # Use train_size instead of test_size for subsampling
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, 
            train_size=subsample,  # Changed from test_size
            random_state=42, 
            stratify=y_train
        )
    else:
        X_sample = X_train.copy()
        y_sample = y_train.copy()
    
    # Fix 2: Updated parameter grid with valid options
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20, None],  # Added None option
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
        "class_weight": ["balanced", "balanced_subsample", None],  # Added None
        "criterion": ["gini", "entropy"]  # Fixed: "log_loss" not valid for RandomForest
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    if rand_search:
        grid_search = RandomizedSearchCV(
            rf, param_grid,
            scoring='roc_auc',
            n_iter=20,
            cv=3,
            n_jobs=-1,
            random_state=42  # Fix 3: Added random_state for reproducibility
        )
    else:
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
    
    # Fit the grid search
    grid_search.fit(X_sample, y_sample)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)
    
    # Get best model and make predictions
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'search_type': 'RandomizedSearchCV' if rand_search else 'GridSearchCV',
        'subsample_used': subsample,
        'all_results': []
    }
    
    # Add all parameter combinations and their scores
    for i, params in enumerate(grid_search.cv_results_['params']):
        results['all_results'].append({
            'params': params,
            'mean_score': grid_search.cv_results_['mean_test_score'][i],
            'std_score': grid_search.cv_results_['std_test_score'][i]
        })
    
    with open(res_name, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {res_name}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_rf_model, grid_search.best_params_

def xgboost_grid_search(X_train, y_train, X_test, y_test, 
                       res_name="xgboost_grid_search.json", 
                       subsample=None, rand_search=True, ):
    """
    Perform grid search for XGBoost with option for subsampling
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data  
    res_name : str, filename for saving results
    subsample : float, fraction of training data to use (e.g., 0.3 for 30%)
    rand_search : bool, whether to use RandomizedSearchCV vs GridSearchCV
    early_stopping : bool, whether to use early stopping in XGBoost
    
    Returns:
    --------
    best_xgb_model : trained model
    best_params : dict of best parameters
    """
    
    # Handle subsampling
    if subsample:
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, 
            train_size=subsample,
            random_state=42, 
            stratify=y_train
        )
        print(f"Using {subsample*100}% of training data: {len(X_sample)} samples")
    else:
        X_sample = X_train.copy()
        y_sample = y_train.copy()
    
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
        )
        
    
    # Define parameter grid for XGBoost
    param_grid = {
        "n_estimators": [ 750],  # More options for boosting
        "learning_rate": [0.01, 0.05, 0.1, 0.2],  # Step size shrinkage
        "max_depth": [3, 4, 5, 6, 8],  # Tree depth
        "subsample": [0.7, 0.8, 0.9, 1.0],  # Row sampling
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],  # Column sampling
        "reg_alpha": [0, 0.01, 0.1, 1],  # L1 regularization
        "reg_lambda": [1, 10, 100],  # L2 regularization
        "gamma": [0, 0.1, 0.5, 1],  # Minimum loss reduction
        "min_child_weight": [1, 3, 5]  # Minimum sum of instance weight
    }
    
    # Create XGBoost classifier with base parameters
    base_params = {
        'random_state': 42,
        'early_stopping_rounds':20
    }
    

    xgb_model = xgb.XGBClassifier(**base_params)
    
    if rand_search:
        print("Running RandomizedSearchCV...")
        grid_search = RandomizedSearchCV(
            xgb_model, param_grid,
            scoring='roc_auc',
            n_iter=30,  # More iterations for XGBoost complexity
            cv=4,  # Fewer folds for speed

            n_jobs=-1,
            random_state=42,
            verbose=1
        )
    else:
        print("Running GridSearchCV...")
        # For GridSearchCV, use smaller parameter grid to avoid excessive runtime
        small_param_grid = {
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 6, 9],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1, 10]
        }
            
        grid_search = GridSearchCV(
            xgb_model,
            small_param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=2
        )
    

    fit_params={
            "eval_set" : [[X_val_sample, y_val_sample]]}
    
    # Custom fit for early stopping (simplified approach)
    grid_search.fit(X_train_sample, y_train_sample, **fit_params)

    
    print("Best parameters:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)
    
    # Get best model and make predictions
    best_xgb_model = grid_search.best_estimator_
    
    # Copy best_params so we don’t mutate sklearn’s object
    best_params = grid_search.best_params_.copy()

    # Overwrite n_estimators with early-stopped value
    best_n_estimators = best_xgb_model.best_iteration + 1
    best_params["n_estimators"] = best_n_estimators
    print("Best n_estimators (from early stopping):", best_n_estimators)
    final_model = xgb.XGBClassifier(
        **grid_search.best_params_,
        random_state=42
    )

    final_model.fit(X_sample, y_sample)

    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)
    # Prepare results
    results = {
        'model_type': 'XGBoost',
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_n_estimators': best_n_estimators, 
        'search_type': 'RandomizedSearchCV' if rand_search else 'GridSearchCV',
        'subsample_used': subsample,
        'feature_importance': best_xgb_model.feature_importances_.tolist(),
        'n_features': X_train.shape[1],
        'all_results': []
    }
    
    # Add all parameter combinations and their scores
    for i, params in enumerate(grid_search.cv_results_['params']):
        results['all_results'].append({
            'params': params,
            'mean_score': grid_search.cv_results_['mean_test_score'][i],
            'std_score': grid_search.cv_results_['std_test_score'][i],
            'rank': int(grid_search.cv_results_['rank_test_score'][i])
        })
    
    # Sort results by rank
    results['all_results'] = sorted(results['all_results'], key=lambda x: x['rank'])
    
    # Save to JSON file
    with open(res_name, 'w') as f:
        json.dump(results, f, indent=4, default=str)  # default=str handles numpy types
    
    print(f"Results saved to {res_name}")
    print(f"Number of features: {X_train.shape[1]}")
    print("\nTop 3 Parameter Combinations:")
    for i, result in enumerate(results['all_results'][:3]):
        print(f"{i+1}. Score: {result['mean_score']:.4f} - {result['params']}")
    
    print("\nFeature Importance (top 5):")
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    importance_pairs = list(zip(feature_names, best_xgb_model.feature_importances_))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    for name, importance in importance_pairs[:5]:
        print(f"  {name}: {importance:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_xgb_model, grid_search.best_params_

def plot_feature_importances(rf_model, X_train, title = "Feature Importance in Random Forest"):
    feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 5))
    feature_importances[:10].plot(kind="bar")  # Show top 10 features
    plt.title(title)
    plt.show()
    return feature_importances
