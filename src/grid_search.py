#!/usr/bin/env python
# encoding: utf-8
import gzip
import json
import math
import pandas as pd
import xgboost as xgb
import numpy as np
import itertools
import argparse
import pickle
from sklearn.metrics import mean_squared_error


def prepare_data():
    print('Preparing data...')
    train_data = pd.read_csv('../data/train.csv')
    # shuffle the train data to create new train data and dev data
    # train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = pd.read_csv('../data/test.csv')
    train_y = train_data.pop('Score')
    # concat train data and test data for global preprocessing
    all_data = pd.concat([train_data, test_data])
    # remove id and sore columns to get X
    all_X = all_data[[col for col in all_data.columns if col not in ['Id', 'Score']]]
    # convert categorical variables to dummy numeric variables
    all_X = pd.get_dummies(all_X)
    # normalize the features
    all_X = all_X / all_X.max()
    # re-split the data into train, dev and test set
    train_X, dev_X = all_X[:int(train_data.shape[0]*0.7)], all_X[int(train_data.shape[0]*0.7):train_data.shape[0]]
    train_y, dev_y = train_y[:int(train_data.shape[0]*0.7)], train_y[int(train_data.shape[0]*0.7): train_data.shape[0]]
    test_X = all_X[train_data.shape[0]:]
    train_X.to_csv('../data/train_data.x.csv', index=False, header=False)
    train_y.to_csv('../data/train_data.y.csv', index=False, header=False)
    dev_X.to_csv('../data/dev_data.x.csv', index=False, header=False)
    dev_y.to_csv('../data/dev_data.y.csv', index=False, header=False)
    test_X.to_csv('../data/test_data.x.csv', index=False, header=False)


def load_data():
    train_X = pd.read_csv('../data/train_data.x.csv').values
    train_y = pd.read_csv('../data/train_data.y.csv').values
    dev_X = pd.read_csv('../data/dev_data.x.csv').values
    dev_y = pd.read_csv('../data/dev_data.y.csv').values
    test_X = pd.read_csv('../data/train_data.x.csv').values
    return train_X, train_y, dev_X, dev_y, test_X


def search_xgb_params(dtrain, ddev):
    # note that the last one in watch list will be used for early stopping
    watch_list = [(dtrain, 'train'), (ddev, 'dev')]
    # use grid search to find the best hyperperameters
    param_grid = {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'max_depth': [2, 3, 5],
        'min_child_weight': [1, 3, 10],
        'subsample': [1],
        'objective': ['reg:linear', 'count:poisson', 'reg:tweedie'],
        'eval_metric': ['rmse'],
        'silent': [True]
    }
    # create all possible hyperperameter groups
    param_groups = itertools.product(*param_grid.values())
    best_params = None
    records = []
    for param_group in param_groups:
        booster_params = {name: value for name, value in zip(param_grid.keys(), param_group)}
        print('Now we are training and validating hyperperameters: {}.'.format(booster_params))
        xgb_model = xgb.train(params=booster_params, dtrain=dtrain, num_boost_round=100000,
                              evals=watch_list, early_stopping_rounds=20, verbose_eval=100)
        if best_params is None or xgb_model.best_score < best_params['rmse']:
            best_params = {'booster_params': booster_params,
                           'num_boost_round': xgb_model.best_iteration + 1,
                           'rmse': xgb_model.best_score}
        records.append({'booster_params': booster_params,
                        'num_boost_round': xgb_model.best_iteration + 1,
                        'rmse': xgb_model.best_score})
    with open('../data/grid_search_records.json', 'w') as fout:
        json.dump(records, fout)
    print('Best parameters are: {} with num_boost_round as {} and rmse as {}'.format(
        best_params['booster_params'],
        best_params['num_boost_round'],
        best_params['rmse']))
    return best_params


def test_xgb_model(train_X, train_y, dev_X, dev_y):
    print('Testing xgb model')
    clf = xgb.XGBRegressor()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(dev_X)
    print('RMSE: {}'.format(math.sqrt(mean_squared_error(dev_y, pred_y))))


def test_knn_model(train_X, train_y, dev_X, dev_y):
    print('Testing knn model...')
    from sklearn.neighbors import KNeighborsRegressor
    clf = KNeighborsRegressor()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(dev_X)
    print('RMSE: {}'.format(math.sqrt(mean_squared_error(dev_y, pred_y))))


def test_svm_model(train_X, train_y, dev_X, dev_y):
    print('Testing svm model...')
    from sklearn.svm import LinearSVR
    clf = LinearSVR()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(dev_X)
    print('RMSE: {}'.format(math.sqrt(mean_squared_error(dev_y, pred_y))))


def test_target_func(train_X, train_y, dev_X, dev_y, target):
    print('Testing target function {}'.format(target))
    if target == 'log':
        train_y = np.log(train_y)
    elif target == 'sqrt':
        train_y = np.sqrt(train_y)
    elif target != 'orig':
        raise ValueError('Unsupported target function {}.'.format(target))
    clf = xgb.XGBRegressor()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(dev_X)
    if target == 'log':
        pred_y = np.power(np.e, pred_y)
    elif target == 'sqrt':
        pred_y == np.power(pred_y, 2)
    elif target != 'orig':
        raise ValueError('Unsupported target function {}.'.format(target))
    print('RMSE: {}'.format(math.sqrt(mean_squared_error(dev_y, pred_y))))


def parse_args():
    parser = argparse.ArgumentParser()
    prepare_group = parser.add_argument_group('preprocessing', 'transform and prepare the data')
    prepare_group.add_argument('--prepare', action='store_true', help='trigger to do data preprocessing.')

    target_group = parser.add_argument_group('target', 'compare which kind of target y.')
    target_group.add_argument('--compare_target', action='store_true', help='trigger to do target comparison')
    target_group.add_argument('--targets', choices=['orig', 'log', 'sqrt'], nargs='*', help='which target to compare.')

    model_compare_group = parser.add_argument_group('model', 'compare different models')
    model_compare_group.add_argument('--compare_model', action='store_true', help='trigger to do model comparison.')
    model_compare_group.add_argument('--models', choices=['xgb', 'knn', 'svm'], nargs='*', help='which model to compare.')

    xgb_test_group = parser.add_argument_group('xgb', 'experiment with xgb model')
    xgb_test_group.add_argument('--xgb', action='store_true', help='trigger to experiment with xgb model')
    xgb_test_group.add_argument('--search', action='store_true', help='whether to find best parameters with grid search')
    xgb_test_group.add_argument('--predict', action='store_true', help='whether to predict labels for test data.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prepare:
        prepare_data()
    if args.compare_target:
        train_X, train_y, dev_X, dev_y, test_X = load_data()
        for target in args.targets:
            test_target_func(train_X, train_y, dev_X, dev_y, target)
    if args.compare_model:
        train_X, train_y, dev_X, dev_y, test_X = load_data()
        if 'xgb' in args.models:
            test_xgb_model(train_X, train_y, dev_X, dev_y)
        if 'knn' in args.models:
            test_knn_model(train_X, train_y, dev_X, dev_y)
        if 'svm' in args.models:
            test_svm_model(train_X, train_y, dev_X, dev_y)
    if args.xgb:
        train_X, train_y, dev_X, dev_y, test_X = load_data()
        if args.search:
            # dtrain = xgb.DMatrix(np.concatenate((train_X, dev_X)), np.concatenate((train_y, dev_y)))
            dtrain = xgb.DMatrix(train_X, train_y)
            ddev = xgb.DMatrix(dev_X, dev_y)
            best_params = search_xgb_params(dtrain, ddev)
            print('Saving parameters...')
            with gzip.open('../data/best_params.pickle.gz', 'wb') as fout:
                pickle.dump(best_params, fout)
        else:
            with gzip.open('../data/best_params.pickle.gz', 'rb') as fin:
                best_params = pickle.load(fin)
        if args.predict:
            dtrain = xgb.DMatrix(np.concatenate((train_X, dev_X)), np.concatenate((train_y, dev_y)))
            dtest = xgb.DMatrix(test_X)
            xgb_model = xgb.train(params=best_params['booster_params'], dtrain=dtrain,
                                  num_boost_round=best_params['num_boost_round'])
            pred_y = xgb_model.predict(dtest)
            with open('../data/predict.txt', 'w') as fout:
                for y in pred_y:
                    fout.write('{}\n'.format(round(y)))

if __name__ == '__main__':
    main()
