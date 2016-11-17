#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import xgboost as xgb
from sklearn.grid_search import GridSearchCV


train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/train.csv')
train_y = train_data.pop('Score')
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
# hyper-parameters that need cross validation
cv_params = {'max_depth': [3, 5],
             'min_child_weight': [1, 3],
             'objective': ['reg:linear', 'count:poisson']}
# fixed hyper-parameters
fixed_params = {'learning_rate': 0.1,
                'n_estimators': 1000,
                'gamma': 0,
                'nthread': -1,
                'seed': 0,
                'subsample':1}
# use grid search to find the best classifier
best_clf = GridSearchCV(xgb.XGBRegressor(**fixed_params), cv_params, scoring='mean_squared_error', cv=2, n_jobs=-1, verbose=10)
best_clf.fit(train_X, train_y)
# log information
print('Grid sores are:')
print(best_clf.grid_scores_)
print('We are now re-train the model using the best hyperparameters: {}'.format(best_clf.best_params_))
final_clf = xgb.XGBRegressor(**best_clf.best_params_)
final_clf.fit(train_X, train_y)
# predict
pred_y = final_clf.predict(test_X)
# output
with open('../data/predict.txt', 'w') as fout:
    for y in pred_y:
        fout.write('{}\n'.format(y))
