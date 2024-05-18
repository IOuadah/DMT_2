# imports

import pandas as pd
import numpy as np
import time
import argparse
import pickle as pkl
import sys
import os
import json

# import the rankers for learning to rank
import lightgbm
import xgboost as xgb
import ptranking
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr

# Additional imports for CatBoost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from copy import deepcopy
import shap

shap.initjs()

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', help='path to the train data csv file')
    parser.add_argument('test_path', help='path to the test data csv file')
    parser.add_argument('out_dir', help='path to an output directory where output is saved')
    parser.add_argument('-r', '--ranker', dest='ranker', choices=['gbm', 'xgb', 'pt', 'tf', 'catboost'], default="gbm")
    args = parser.parse_args()
    return args

# Loading data
def load_data(path):
    print("----------Loading the data----------\n")
    data = pd.read_csv(path)
    return data

# Adding time data features
def add_time_data(data):
    print("----------Adding time data----------\n")
    date_time = pd.to_datetime(data.pop("date_time"))
    data["year"] = date_time.dt.year
    data["month"] = date_time.dt.month
    data["day"] = date_time.dt.dayofweek
    return data

# Adding custom features
def add_features(data):
    print("----------Adding new features/columns----------\n")
    data["avg_location_score"] = data[['prop_location_score1', 'prop_location_score2']].mean(axis=1)
    data = data.drop(['prop_location_score1', 'prop_location_score2'], axis=1)
    data['family'] = (data['srch_children_count'] > 0).astype(int)
    data['total_price_stay_sqrt'] = np.sqrt((data.pop('price_usd') * data['srch_length_of_stay']))
    return data

# Removing columns with high null values
def remove_null(data, user_info, metrics):
    print("----------Removing columns with NULL > 50%----------\n")
    skip = user_info + metrics
    null_percent = (data.isna().sum() * 100 / len(data))
    nulls_to_drop = null_percent[null_percent > 50].index.tolist()
    nulls_to_drop = [null for null in nulls_to_drop if null not in skip]
    data_processed = data.drop(nulls_to_drop, axis=1)
    return data_processed

# Preprocessing data
def preprocess_data(data, ranker, query, metrics, user_info, train=True):
    print("----------Starting Pre-Processing----------\n")
    if train:
        conditions = [data["booking_bool"] == 1, data["click_bool"] == 1]
        scores = [2, 1]
        data["target_score"] = np.select(conditions, scores, 0)
        metrics = metrics + ["target_score"]
    
    data = add_time_data(data)
    data = remove_null(data, user_info=user_info, metrics=metrics)
    data = add_features(data)
    data.sort_values(by=query, inplace=True)
    data.set_index(query, inplace=True)
    
    if not train:
        return data
    
    features = data.drop(metrics, axis=1)
    X, y = features, data["target_score"].values
    return X, y

# Splitting the data
def val_split(X_data, y_data, val_fraction):
    print("----------Splitting the data----------\n")
    val_size = int(val_fraction * len(X_data))
    X_train, X_val = X_data[:-val_size], X_data[-val_size:]
    y_train, y_val = y_data[:-val_size], y_data[-val_size:]
    return X_train, y_train, X_val, y_val

# Training the CatBoost model
def train_catboost(X_train, y_train, X_val, y_val, output_dir, categorical_cols):
    print("----------Training the CatBoost model----------\n")
    train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_cols, group_id=X_train.index)
    eval_pool = Pool(data=X_val, label=y_val, cat_features=categorical_cols, group_id=X_val.index)
    
    parameters = {
        'iterations': 2000,
        'custom_metric': ['NDCG', "AUC:type=Ranking"],
        'verbose': False,
        'random_seed': 42,
        "has_time": True,
        "metric_period": 4,
        "save_snapshot": False,
        "use_best_model": True,
        'loss_function': 'YetiRank'
    }
    
    model = CatBoost(parameters)
    model.fit(train_pool, eval_set=eval_pool, plot=True)
    
    pkl.dump(model, open(os.path.join(output_dir, "catboost_model.dat"), "wb"))
    return model

# Prediction function
def predict(model, test_data, output_dir):
    print("----------Making the predictions----------\n")
    test_data = test_data.copy().reset_index()
    submission = test_data[["srch_id", "prop_id"]]
    predictions = model.predict(test_data)
    submission["prediction"] = predictions
    submission = submission.sort_values(["prediction"], ascending=False)
    submission[["srch_id", "prop_id"]].to_csv(os.path.join(output_dir, "submission.csv"), index=False)

def main():
    args = parse_args()

    user_info = ["visitor_location_country_id", "visitor_hist_starrating", "visitor_hist_adr_usd"]
    metrics = ["click_bool", "booking_bool", "gross_bookings_usd"]
    query = "srch_id"
    ranker = args.ranker
    output_dir = args.out_dir

    train_data = load_data(args.train_path)
    X_train, y_train = preprocess_data(train_data, ranker=ranker, query=query, metrics=metrics, user_info=user_info)
    X_train, y_train, X_val, y_val = val_split(X_train, y_train, val_fraction=0.3)
    
    if ranker == 'catboost':
        categorical_cols = ['prop_id', "srch_destination_id", "year", "month", "day"]
        model = train_catboost(X_train, y_train, X_val, y_val, output_dir, categorical_cols)
    else:
        # Handle other rankers like gbm, xgb, tf, pt
        pass

    test_data = load_data(args.test_path)
    test_data = preprocess_data(test_data, ranker=ranker, query=query, metrics=metrics, user_info=user_info, train=False)

    predict(model, test_data, output_dir)

if __name__ == "__main__":
    main()


