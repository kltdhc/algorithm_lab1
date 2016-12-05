# algorithm_lab1
A Rating Task


## Reference 

1. Use xgboost with grid search and cross validation. [https://jessesw.com/XG-Boost/](https://jessesw.com/XG-Boost/)
2. XGBoost parameter settings. [http://blog.csdn.net/wzmsltw/article/details/50994481](http://blog.csdn.net/wzmsltw/article/details/50994481)

## Some Results:
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
 