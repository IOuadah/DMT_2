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


def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."

    parser = argparse.ArgumentParser("i tried")

    parser.add_argument('train_path', help='path to the train data csv file')
    parser.add_argument('test_path', help='path to the test datat csv file')

    parser.add_argument('out_dir',
                        help='path to an output directory where output is saved')
    parser.add_argument('-r', '--ranker', dest='ranker',
                        choices=['gbm', 'xgb', 'pt', 'tf'], default="gbm")

    args = parser.parse_args()

    return args


# loading data
def load_data(path):
    print("----------Loading the data----------\n")
    data = pd.read_csv(path)
    return data

def add_time_data(data):
    print("----------Adding time data----------\n")
    date_time = pd.to_datetime(data.pop("date_time"))
    data["year"] = date_time.dt.year
    data["month"] = date_time.dt.month
    data["day"] = date_time.dt.dayofweek

    return data
    

def add_features(data):
    print("----------Adding new features/columns----------\n")
    data["avg_location_score"] = data[['prop_location_score1', 'prop_location_score2']].mean(axis=1)
    data = data.drop(['prop_location_score1', 'prop_location_score2'],axis = 1)

    data['family'] = (data['srch_children_count'] > 0).astype(int) 

    # data['total_price_stay'] = data['price_usd'] * data['srch_length_of_stay']
    data['total_price_stay_sqrt'] = np.sqrt((data.pop('price_usd') * data['srch_length_of_stay']))
    return data

def remove_null(data, user_info, metrics):
    print("----------Removing columns with NULL > 50%----------\n")
    # skip these features cause the null had a meaning 
    skip = user_info + metrics
    
    null_percent = (data.isna().sum() * 100/len(data))

    nulls_to_drop = null_percent[null_percent > 50].index.tolist()
    nulls_to_drop = [null for null in nulls_to_drop if null not in skip]

    data_processed = data.drop(nulls_to_drop,axis=1)
    return data_processed

def normalize(data):
    print("----------Normalizing the data----------\n")
    data = (data - data.mean()) / data.std()
    return data

# def add_target():
#     pass

def preprocess_data(data, ranker, query, metrics, user_info, train = True):
    print("----------Starting Pre-Processing----------\n")
    print(data.columns)
    if train:
        conditions = [data["booking_bool"] == 1, data["click_bool"] == 1]
        scores = [5, 1]
        data["target_score"] = np.select(conditions, scores, 0)
        metrics = metrics + ["target_score"]
    
    data = add_time_data(data)

    data = remove_null(data, user_info=user_info, metrics=metrics)

    # data = normalize(data)
    data = add_features(data)


    data.sort_values(by=query, inplace=True)
    data.set_index(query, inplace=True)
    
    if train == False:
        return data
    


    features = data.drop(metrics, axis=1)
    X, y = features, data["target_score"].values

    if ranker == 'pt':
        # from ptranking.eval.parameter import DataSetting, EvalSetting, ModelParameter, ScoringFunctionParameter
        pass
    
    return X, y

def val_split(X_data, y_data, val_fraction):
    print("----------Splitting the data----------\n")
    val_size = int(val_fraction * len(X_data))
    X_train, X_val = X_data[:-val_size], X_data[-val_size:]
    y_train, y_val = y_data[:-val_size], y_data[-val_size:]

    return X_train, y_train, X_val, y_val

def train_model(X_train, y_train, X_val, y_val, ranker, query, out_dir, learning_rate=0.12, boost_method="dart"):
    # print(type(X_train))

    def get_cat_cols(data):
        cat_features = ["site_id", 'visitor_location_country_id', 'prop_country_id', 
                        'srch_destination_id', "year", "month", "day"]
        cat_features_numbers = [data.columns.get_loc(cat) for cat in cat_features if cat in data.columns]
        return cat_features_numbers
    
    cat_features_indx = get_cat_cols(X_train)


    print("----------Training the model----------\n")  
    if ranker == 'kmeans':
        # kmeans = KMeans(n_clusters=5)
        # kmeans.fit(train_data)
        pass

    elif ranker == 'gbm':
        def get_group_size(data):
            g_size = data.reset_index().groupby(query)[query].count().tolist()
            return g_size
        
        group_size_train = get_group_size(X_train)
        group_size_val = get_group_size(X_val)

        model = lightgbm.LGBMRanker(objective="lambdarank", metric="ndcg@5", learning_rate=0.01, 
                                    n_estimators=1024,  boosting=boost_method, )
        # model.fit(X_train, y_train, group=group_size_train, eval_set=[(X_val, y_val)], eval_group=[group_size_val],
        #           eval_metric=['ndcg@5'])


        # model = lightgbm.LGBMRanker(objective="lambdarank", metric="ndcg@5", n_estimators=512, learning_rate=learning_rate, 
        #                             label_gain=[0, 1, 2], seed=42, boosting=boost_method)
        
        model.fit(X_train, y_train, group=group_size_train, eval_set=[(X_val, y_val)], eval_group=[group_size_val],
                  eval_metric=['ndcg@5'], categorical_feature = cat_features_indx)


    elif ranker == 'xgb':

        qid_t = X_train.copy().reset_index()
        qid_train = qid_t.srch_id
        qid_v = X_val.copy().reset_index()
        qid_val = qid_v.srch_id

        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()

        model = xgb.XGBRanker(
        n_estimators=5000,
        tree_method="hist",
        device="cuda",
        learning_rate=0.01,
        reg_lambda=1.5,
        subsample=0.8,
        sampling_method="gradient_based",
        # LTR specific parameters
        objective="rank:ndcg",
        # - Enable bias estimation
        lambdarank_unbiased=True,
        # - normalization (1 / (norm + 1))
        lambdarank_bias_norm=1,
        # - Focus on the top 12 documents
        lambdarank_num_pair_per_sample=13,
        lambdarank_pair_method="topk",
        ndcg_exp_gain=True,
        eval_metric=["ndcg@5"]
    )

        model.fit(X_train, y_train, qid=qid_train, eval_set=[(X_val, y_val)], eval_qid=[qid_val], verbose=True)

    elif ranker == 'pt':
        # ltr_evaluator = ptranking.ltr_adhoc.eval.ltr.LTREvaluator()
        # debug = True

        # ltr_evaluator.set_eval_setting(debug=debug, dir_output=out_dir)
        # ltr_evaluator.set_data_setting(debug=debug, data_id=None, dir_data=data)
        # data_dict = ltr_evaluator.get_default_data_setting()
        # eval_dict = ltr_evaluator.get_default_eval_setting()

        # ltr_evaluator.set_model_setting(debug=debug, model_id='LambdaRank', data_dict=data_dict) # data_dict argument is required
        # model_para_dict = ltr_evaluator.get_default_model_setting()

        # ''' basic check before loading the ranker '''
        # ltr_evaluator.setup_eval(data_dict=data_dict, eval_dict=eval_dict, 
        #                          sf_para_dict=sf_para_dict, model_para_dict=model_para_dict)

        # ''' initialize the ranker '''
        # model = ltr_evaluator.ltr_evaluator.load_ranker(sf_para_dict=sf_para_dict, model_para_dict=model_para_dict)

        # losses, train_ndcgs, test_ndcgs = ltr_evaluator.ltr_evaluator.naive_train(ranker=model, eval_dict=eval_dict, 
        #                                                                           train_data=train_data, test_data=test_data)
        pass
        
    elif ranker == 'tf':
        train_ds = ...

        inputs = {"float_features": tf.keras.Input(shape=(None, 136), dtype=tf.float32)}
        norm_inputs = [tf.keras.layers.BatchNormalization()(x) for x in inputs.values()]
        x = tf.concat(norm_inputs, axis=-1)
        for layer_width in [128, 64, 32]:
            x = tf.keras.layers.Dense(units=layer_width)(x)
            x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)
            scores = tf.squeeze(tf.keras.layers.Dense(units=1)(x), axis=-1)

        model = tf.keras.Model(inputs=inputs, outputs=scores)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                        loss=tfr.keras.losses.SoftmaxLoss(),
                        metrics=tfr.keras.metrics.get("ndcg", topn=5, name="NDCG@5"))
        model.fit(train_ds, epochs=3)
    else:
        raise ValueError('Ranker not supported')
    
    pkl.dump(model, open(os.path.join(out_dir, "model.dat"), "wb"))

    return model

def predict(test_data, output_dir):
    print("----------Making the predictions----------\n")
    model = pkl.load(open(os.path.join(output_dir, "model.dat"), "rb"))

    test_data = test_data.copy().reset_index()

    submission = test_data[["srch_id", "prop_id"]]

    # test_data = remove_columns(test_data)

    # categorical_features_numbers = get_categorical_column(test_data)

    # print("Predicting on train set with columns: {}".format(test_data.columns.values))
    # kwargs = {}
    # kwargs = {"categorical_feature": categorical_features_numbers}

    predictions = model.predict(test_data)
    submission["prediction"] = predictions
    del test_data

    submission = submission.sort_values([ "prediction"], ascending=False)
    print(submission.prediction)
    submission[["srch_id", "prop_id"]].to_csv(os.path.join(output_dir, "submission.csv"), index=False)


def main():
    args = parse_args()

    user_info = ["visitor_location_country_id", "visitor_hist_starrating", "visitor_hist_adr_usd"]    
    metrics = ["click_bool", "booking_bool", "gross_bookings_usd"]
    query = "srch_id"

    ranker = args.ranker

    output_dir = args.out_dir
    # output_dir = "output"
    train_data = load_data(args.train_path)
    # train_data = load_data("data/train_small.csv")


    X_train, y_train = preprocess_data(train_data, ranker=ranker, query=query, metrics=metrics, user_info=user_info)

    X_train, y_train, X_val, y_val = val_split(X_train, y_train, val_fraction=0.3)

    model = train_model(X_train, y_train, X_val, y_val, ranker, query, output_dir)

    test_data = load_data(args.test_path)
    test_data = preprocess_data(test_data, ranker=ranker, query=query, metrics=metrics, user_info=user_info, train=False)

    # test_data = load_data("data/train_small.csv")
    predict(test_data, output_dir)

if __name__ == "__main__":
    main()
