#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import xgboost as xgb
import numpy as np
import itertools
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='use log value as the target.')
    parser.add_argument('--sqrt', action='store_true', help='use sqrt value as the target.')
    return parser.parse_args()

def main():
    args = parse_args()
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/train.csv')
    train_y = train_data.pop('Score')
    if args.log:
        train_y = np.log(train_y)
    if args.sqrt:
        train_y = np.sqrt(train_y)
    # concat train data and test data for global preprocessing
    all_data = pd.concat([train_data, test_data])
    all_X = all_data[[col for col in all_data.columns if col not in ['Id', 'Score']]]
    # convert categorical variables to dummy numeric variables
    all_X = pd.get_dummies(all_X)
    # normalize the features
    all_X = all_X / all_X.max()
    # re-split the data into train and test
    train_X = all_X[:train_data.shape[0]]
    test_X = all_X[train_data.shape[0]:]
    dtrain = xgb.DMatrix(train_X, train_y)
    dtest = xgb.DMatrix(test_X)
    # use grid search to find the best hyperperameters
    param_grid = {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [2, 3, 5, 10],
        'min_child_weight': [1, 5, 10, 30],
        'subsample': [1, 0.7],
        'objective': ['reg:tweedie', 'reg:linear', 'count:poisson'],
        'silent': [True]
    }
    # create all possible hyperperameter groups
    param_groups = itertools.product(*param_grid.values())

    best_booster_params = None
    for param_group in param_groups:
        booster_params = {name: value for name, value in zip(param_grid.keys(), param_group)}
        print('Now we are training and validating hyperperameters: {}.'.format(booster_params))
        result = xgb.cv(params=booster_params, num_boost_round=10000, early_stopping_rounds=5, dtrain=dtrain, nfold=3, metrics='rmse')
        print('The cross validation result for each boosting round is: \n{}'.format(result))
        last_iter_result = result.iloc[-1]
        num_boost_round = last_iter_result.name
        rmse = last_iter_result['test-rmse-mean']
        if best_booster_params is None or rmse < best_booster_params['rmse']:
            best_booster_params = {'booster_params': booster_params, 'num_boost_round': num_boost_round, 'rmse': rmse}

    print('Best parameters are: {} with num_boost_round as {} and rmse as {}'.format(best_booster_params['booster_params'],
                                                                                    best_booster_params['num_boost_round'],
                                                                                    best_booster_params['rmse']))
    # train the final model using the best hyperperameters
    final_model = xgb.train(params=best_booster_params['booster_params'], dtrain=dtrain, num_boost_round=best_booster_params['num_boost_round'])

    # predict
    pred_y = final_model.predict(dtest)
    if args.log:
        pred_y = np.power(np.e, pred_y)
    if args.sqrt:
        pred_y = np.power(pred_y, 2)
    # output
    with open('../data/predict.txt', 'w') as fout:
        for y in pred_y:
            fout.write('{}\n'.format(y))

if __name__ == '__main__':
    main()
