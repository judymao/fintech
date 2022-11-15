# Import relevant packages
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve, plot_roc_curve, plot_precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from utils import process_data, compare_models, grid_search, get_gridsearch_results


TRAIN_TREE = False
TRAIN_RF = False

def train_tree_models(X_train, y_train, verbose=False):

    if TRAIN_TREE:
        print("Training decision tree model ...")
        
        # Specify scoring criterion
        criterion = make_scorer(roc_auc_score, needs_proba=True)
        clf_tree = DecisionTreeClassifier()

        param_grid = dict(criterion=['gini', 'entropy'],
                          max_depth=[2, 5, 10],
                          ccp_alpha=[0, 0.00001, 0.0001])
        
        grid_result_tree = grid_search(clf_tree, criterion, param_grid, k=4, X=X_train, y=y_train)

        # Choose best parameters from hyperparameter tuning
        clf_tree = grid_result_tree.best_estimator_

        print("Finished training decision tree model. Saving results.")
        
        # save best decision tree model
        pickle.dump(clf_tree, open('models/model_tree.sav', 'wb'))
        
        if verbose:
            print(get_gridsearch_results(grid_result_tree))
        
    if TRAIN_RF:
        print("Training random forest model ...")
        
        # Specify scoring criterion
        criterion = make_scorer(roc_auc_score, needs_proba=True)

        clf_rf = RandomForestClassifier()

        param_grid = dict(criterion=['entropy'],
                          max_depth=[2, 5, 10],
                          ccp_alpha=[0, 1e-5, 1e-4],
                          n_estimators=[100, 150, 200]
                          )

        grid_result_rf = grid_search(clf_rf, criterion, param_grid, k=4, X=X_train, y=y_train)

        # Choose best parameters from hyperparameter tuning
        clf_rf = grid_result_rf.best_estimator_
        
        print("Finished training random forest model. Saving results.")
        
        # save best decision tree model
        pickle.dump(clf_rf, open('models/model_rf.sav', 'wb'))
        
        if verbose:
            print(get_gridsearch_results(grid_result_rf))

if __name__ == "__main__":
    # Read in data
    data = process_data(type_ = 'normal')
    
    X_train = data['X_train_scaled']
    y_train = data['y_train']

    X_test = data['X_test_scaled']
    y_test = data['y_test']
    
    # Run training
    train_tree_models(X_train, y_train)

    # read best decision tree model
    clf_tree = pickle.load(open('models/model_tree.sav', 'rb'))
    
    # read best rf model
    clf_rf = pickle.load(open('models/model_rf.sav', 'rb'))
    
    clfs = [clf_tree, clf_rf]
    clf_names = ['decision_tree', 'random_forest']
    
    print(compare_models(clfs, clf_names, X_train, y_train, X_test, y_test))