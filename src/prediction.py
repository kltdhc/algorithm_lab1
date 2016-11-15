import xgboost as xgb

dtrain = xgb.DMatrix('../data/data.txt.train')
dtest = xgb.DMatrix('../data/data.txt.test')

param = {
    'max_depth': 10,
    'eta': 1.0,
    'gamma': 1.0,
    'min_child_weight': 1,
    'save_period': 0,
    'objective': 'reg:linear'
}

num_round = 5
watch_list = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, watch_list)
preds = bst.predict(dtest)
print(preds)
