import pandas as pd
import numpy as np
import xgboost as xgb
from hyperopt import fmin, tpe, hp, space_eval
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

global dtrain


def load_data(path):  # split data into X and Y
    csv_file = pd.read_csv(path)
    data_set = csv_file.values
    _x = data_set[0:40000, 2:34]
    _y = data_set[0:40000, 1]
    return _x, _y


def one_hot_coding(x):  # if feature is categorical variable, do OneHot coding
    _encoded_x = None
    for i in range(0, x.shape[1]):
        feature = x[:, i]
        if isinstance(feature[0], str):
            _label_encoder = LabelEncoder()
            feature = _label_encoder.fit_transform(x[:, i])
            feature = feature.reshape(x.shape[0], 1)
            _oneHotEncoder = OneHotEncoder(sparse=False)
            feature = _oneHotEncoder.fit_transform(feature)
        else:
            feature = feature.reshape(x.shape[0], 1)
        if _encoded_x is None:
            _encoded_x = feature
        else:
            _encoded_x = np.concatenate((_encoded_x, feature), axis=1)
    return _encoded_x


def objective(args):  # least rmse
    global dtrain
    _params = args
    _result = xgb.cv(params=_params, dtrain=dtrain, num_boost_round=1000,
                     early_stopping_rounds=5, nfold=3, metrics='rmse')
    _lastIterResult = _result.iloc[-1]
    _rmse = _lastIterResult['test-rmse-mean']
    return _rmse


def find_best_params():  # use hyperopt to find the best args
    global dtrain
    params = {
        'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
        'max_depth': hp.choice('max_depth', range(1, 9)),
        'max_delta_step': hp.choice('max_delta_step', [0.1 * i for i in range(0, 11)]),
        'eta': hp.choice('eta', [0.1 * i for i in range(8, 13)]),
        'learning_rate': hp.choice('learning_rate', [0.1 * i for i in range(1, 3)]),
        'gamma': hp.choice('gamma', [0.1 * i for i in range(8, 13)]),
        'subsample': hp.choice('subsample', [0.1 * i for i in range(8, 11)]),
        'min_child_weight': hp.choice('min_child_weight', [0.1 * i for i in range(8, 13)]),
        'save_period': 0,
        'objective': hp.choice('objective', ['reg:linear', 'count:poisson']),
        'eval_metric': 'rmse'
    }
    best = fmin(objective, params, algo=tpe.suggest, max_evals=3)
    best_params = space_eval(params, best)
    result = xgb.cv(params=best_params, dtrain=dtrain, num_boost_round=200,
                    early_stopping_rounds=5, nfold=3, metrics='rmse')
    last_iter_result = result.iloc[-1]
    num_boost_round = last_iter_result.name
    rmse = last_iter_result['test-rmse-mean']
    print('Search done, the best paramss are:')
    print(best_params)
    print('The best numRound is:{}'.format(num_boost_round))
    print('The best rmse is: %.6f.' % rmse)
    return best_params, num_boost_round


def predict(dTest, final_model):  # write the prediction to file
    pred_y = final_model.predict(dTest)
    with open('./output/predict.csv', 'w') as fout:
        fout.write('Id,Score\n')
        i = 40000
        for y in pred_y:
            i += 1
            fout.write('{},{}\n'.format(i, int(y+0.5)))
    print('Prediction done.')


def main():
    global dtrain
    x, y = load_data('./input/train.csv')
    encoded_x = one_hot_coding(x)
    y = y.reshape(y.shape[0], 1)
    dtrain = xgb.DMatrix(encoded_x, y)
    csvtest = pd.read_csv('./input/test.csv').values[:, 1:33]
    csvtest = one_hot_coding(csvtest)
    dtest = xgb.DMatrix(csvtest)
    #best_params, best_num_round = find_best_params()
    best_params = {
        'max_depth': 3,
        'max_delta_step': 0.2,
        'eta': 0.8,
        'learning_rate': 0.12,
        'gamma': 1.1,
        'subsample': 1,
        'min_child_weight': 1,
        'save_period': 0,
        'objective': 'count:poisson',
        'eval_metric': 'rmse'
    }
    best_num_round = 200
    '''
    # randomly choose 80% as training set
    x_train, x_test, y_train, y_test = train_test_split(encoded_x, y, test_size=0.2)
    xgtrain = xgb.DMatrix(x_train, y_train)
    xgval = xgb.DMatrix(x_test, y_test)
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]
    finalmodel = xgb.train(best_params, xgtrain, best_num_round, watchlist)
    '''
    finalmodel = xgb.train(best_params, dtrain, best_num_round)
    print('Training done.')
    predict(dtest, finalmodel)

if __name__ == '__main__':
    main()





