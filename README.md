# algorithm_lab1
A Rating Task


## Reference 

1. Use xgboost with grid search and cross validation. [https://jessesw.com/XG-Boost/](https://jessesw.com/XG-Boost/)
2. XGBoost parameter settings. [http://blog.csdn.net/wzmsltw/article/details/50994481](http://blog.csdn.net/wzmsltw/article/details/50994481)

## Some Results:

* Train data average: 4.01102555

* Compare different models:

model | parameters | rmse 
 ------   | -------------- | -------
 knn     | default          | 4.126
 linear-svm    | default | 3.936
 xgb-regressor | default | 3.726
 
* Compare different targets:
 
 target | model | rmse
 -------  | ------ | -------
 orig | xgb | 3.726
 sqrt | xgb | 3.973
 log | xgb | 4.445
 
 * Test cascade prediction:
 
Totally, there are 27999 instances for training, while 2564 of them has target value >= 10.

So, it's very imbalanced. We use xgboost to predict whether the target value is >= 10. 
But only 17 of 11999 instances are predicted as true on the dev set.
And 42 of 27999 instances are predicted as true on the train set.
On dev set, the precision score of the prediction is 0.588.
With cascade prediction, the rmse is 3.7255, slightly better than the rmse without it, 3.7258.