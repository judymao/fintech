import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import StandardScaler

BATCH_NUM = 19

def read_batched_data(batches = BATCH_NUM):
    print("Reading in batched data ...")
    data_arr = []
    for i in tqdm(range(batches)):
        data_arr.append(pd.read_csv(f'PS_processed_data_batch{i+1}.csv'))
    data = pd.concat(data_arr)
    data = data.drop(columns=['Unnamed: 0'])
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