import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import plot_confusion_matrix, auc, roc_curve, plot_roc_curve, plot_precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

BATCH_NUM = 19

def read_batched_data(batches = BATCH_NUM):
    print("Reading in batched data ...")
    data_arr = []
    for i in tqdm(range(batches)):
        data_arr.append(pd.read_csv(f'PS_processed_data_batch{i+1}.csv'))
    data = pd.concat(data_arr)
    
    # TBU: dropping 'step' for now
    data = data.drop(columns=['Unnamed: 0','step'])
    return data

def scale_data(X_train, X_test):
    print("Scaling data ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
    
def split_data(X, y, test_size):
    print("Splitting data into train and test ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test

def split_train(X, y, keep_percent):
    print(f"Keep {keep_percent*100}% train data ...")
    X_keep, X_discard, y_keep, y_discard = train_test_split(X, y, test_size=1-keep_percent, random_state=1, stratify=y)
    return X_keep, y_keep

def split_data_time(X, y, folds):
    print(f"Splitting data into {folds} folds of train and test ...")
    tscv = TimeSeriesSplit(n_splits = folds)
    time_series_kfold = []
    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index,:]
        X_test = X.iloc[test_index,:]
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        time_series_kfold.append({
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y[train_index],
            'y_test': y[test_index],
            'scaler': scaler,
        })
    return time_series_kfold

def process_data(type_, folds = 5):
    if type_ not in ['normal']:
        raise ValueError('Type not accepted! Time series to be implemented!')
    
    print("Beginning data processing ...")
    all_data = read_batched_data()
    X, y = all_data[[col for col in all_data.columns if col != 'isFraud']], all_data['isFraud']
    
    if type_ == 'normal':
        X_train, X_test, y_train, y_test = split_data(X, y, 1/folds)
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
        print(f"Completed {type_} data processing.")
        return {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
        }
    else:
        print(f"Completed {type_} data processing.")
        return None
    
# compare final models
def compare_models(models, model_names, X_train, y_train, X_test, y_test):
    compare_results = {}
    for i in range(len(models)):
        model = models[i]
        model_results = {}
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        model_results['accuracy_train'] = accuracy_score(y_train, y_pred_train)
        model_results['recall_train'] = recall_score(y_train, y_pred_train)
        model_results['precision_train'] = precision_score(y_train, y_pred_train)
        model_results['roc_auc_train'] = roc_auc_score(y_train, model.predict_proba(X_train)[:,1])
        model_results['accuracy_test'] = accuracy_score(y_test, y_pred_test)
        model_results['recall_test'] = recall_score(y_test, y_pred_test)
        model_results['precision_test'] = precision_score(y_test, y_pred_test)
        model_results['roc_auc_test'] = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    
        compare_results[model_names[i]] = model_results
    
    return pd.DataFrame(compare_results)

def grid_search(model, criterion, param_grid, k, X, y, verbose=True):
    # Testing through a 5-fold CV and finding the combination that yields the highest criterion
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        scoring=criterion,
                        verbose=10*verbose,
                        cv=StratifiedKFold(n_splits=k),
                        n_jobs=1)

    grid_result = grid.fit(X, y)
    return grid_result

def get_gridsearch_results(grid_search_result):
    res = {}
    
    param_names = list(grid_search_result.cv_results_['params'][0].keys())
    for param in param_names:
        params = grid_search_result.cv_results_['param_' + param]
        res[param] = params
    
    res['criterion_result'] = grid_search_result.cv_results_['mean_test_score']
    return pd.DataFrame(res)