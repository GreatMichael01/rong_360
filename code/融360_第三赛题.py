# !/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@file: predata_upMySQL.py
@company: FInSight
@author: Zhao Ming
@time: 2019-04-10   10:00:10
"""


import numpy as np
import pandas as pd
import time
from datetime import datetime
import xgboost as xgb


"""读取训练集数据"""
orig_train1 = pd.read_tabel("../data/train/train_1.txt")
orig_train2 = pd.read_tabel("../data/train/train_2.txt", names=list(orig_train1.columns.values))
orig_train3 = pd.read_tabel("../data/train/train_3.txt", names=list(orig_train1.columns.values))
orig_train4 = pd.read_tabel("../data/train/train_4.txt", names=list(orig_train1.columns.values))
orig_train5 = pd.read_tabel("../data/train/train_5.txt", names=list(orig_train1.columns.values))
orig_train = pd.concat([orig_train1, orig_train2, orig_train3, orig_train4, orig_train5], axis=0)

"""读取测试集数据"""
train_column_list = list(orig_train.columns.values)
train_column_list.remove("label")
train_column_list.remove("tag")
orig_test = pd.read_table("../data/test/test.txt")[train_column_list]


"""建立模型"""
# 3.1模型一
train = orig_train.copy()
test = orig_test.copy()
train["weekday"] = train.loan_dt.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
test["weekday"] = test.loan_dt.apply(lambda x: datetime.strptime(x, "%Y-%m-%d").weekday())
data_train = train.drop(["loan_dt", "tag", "id"], axis=1)
data_test_drop_label = test.drop(["loan_dt"], axis=1)
data_train_drop_label = data_train.drop("label", axis=1)
data_test = data_test_drop_label.drop(["id"], axis=1)

data_test_id = data_test_drop_label.id
y = data_train.label
# model_1------388
params1 = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eval_metric": "auc",
          "verbose": 1,
          "eta": 0.01,
          "max_delta_step": 20,
          "max_depth": 20,
          "alpha": 10,
          "lambda": 10,
          "gamma": 2,
          "n_jobs": -1}

data_train_matrix = xgb.DMatrix(data_train_drop_label, y)
data_test_matrix = xgb.DMatrix(data_test)
model1 = xgb.train(params1, data_train_matrix, num_boost_round=3000)
predict1 = model1.predict(data_test_matrix)

# 3.2模型二
# model_2------049
data_train = train.drop(["loan_dt", "tag", "id"], axis=1)
data_test_drop_label = test.drop(["loan_dt"], axis=1)
data_train_drop_label = data_train.fillna(value=-99999).drop("label", axis=1)
data_test = data_test_drop_label.fillna(value=-99999).drop(["id"], axis=1)
data_test_id = data_test_drop_label.id
y = data_train.label
params2 = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eval_metric": "auc",
          "verbose": 1,
          "eta": 0.06,
          "max_delta_step": 20,
          "max_depth": 20,
          "alpha": 10,
          "lambda": 40,
          "n_jobs": -1}
data_train_matrix = xgb.DMatrix(data_train_drop_label, y)
data_test_matrix = xgb.DMatrix(data_test)
model2 = xgb.train(params2, data_train_matrix, num_boost_round=800)
predict2 = model2.predict(data_test_matrix)

# 3.3模型三
# model_3------500
data_train = train.drop(["loan_dt", "tag", "id"], axis=1)
data_test_drop_label = test.drop(["loan_dt"], axis=1)
data_train_drop_label = data_train.drop("label", axis=1)
data_test = data_test_drop_label.drop(["id"], axis=1)
data_test_id = data_test_drop_label.id
y = data_train.label
params3 = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eval_metric": "auc",
          "verbose": 1,
          "eta": 0.01,
          "max_delta_step": 10,
          "max_depth": 20,
          "alpha": 10,
          "lambda": 10,
          "n_jobs": -1}
data_train_matrix = xgb.DMatrix(data_train_drop_label, y)
data_test_matrix = xgb.DMatrix(data_test)
model3 = xgb.train(params3, data_train_matrix, num_boost_round=800)
predict3 = model3.predict(data_test_matrix)


# 模型四
# model4-----204
data_train = train.drop(["loan_dt", "tag", "id"], axis=1)
data_test_drop_label = test.drop(["loan_dt"], axis=1)
data_train_drop_label = data_train.drop("label", axis=1)
data_test = data_test_drop_label.drop(["id"], axis=1)
data_test_id = data_test_drop_label.id
y = data_train.label
params4 = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eval_metric": "auc",
          "verbose": 1,
          "eta": 0.01,
          "max_delta_step": 20,
          "max_depth": 20,
          "alpha": 10,
          "lambda": 10,
          "gamma": 2,
          "subsample": 0.8,
          "colsample_bytlevel": 0.8,
          "n_jobs": -1}
data_train_matrix = xgb.DMatrix(data_train_drop_label, y)
data_test_matrix = xgb.DMatrix(data_test)
model4 = xgb.train(params4, data_train_matrix, num_boost_round=3000)
predict4 = model4.predict(data_test_matrix)


# 模型五
# model_5-----418
data_train = train.drop(["loan_dt", "tag", "id"], axis=1)
data_test_drop_label = test.drop(["loan_dt"], axis=1)
data_train_drop_label = data_train.fillna(value=-99999).drop("label", axis=1)
data_test = data_test_drop_label.fillna(value=-99999).drop(["id"], axis=1)
data_test_id = data_test_drop_label.id
y = data_train.label
params5 = {"booster": "gbtree",
          "objective": "binary:logistic",
          "eval_metric": "auc",
          "verbose": 1,
          "eta": 0.01,
          "max_delta_step": 20,
          "max_depth": 20,
          "alpha": 10,
          "lambda": 10,
          "gamma": 2,
          "subsample": 0.8,
          "colsample_bytlevel": 0.8,
          "min_child_weight": 5,
          "n_jobs": -1}
data_train_matrix = xgb.DMatrix(data_train_drop_label, y)
data_test_matrix = xgb.DMatrix(data_test)
model5 = xgb.train(params5, data_train_matrix, num_boost_round=3000)
predict5 = model5.predict(data_test_matrix)

# 3.6模型融合
prob1 = 0.5*(0.5*(0.5*predict1 + 0.5*predict2) + 0.5*predict3) + 0.5*predict4
prob2 = 0.5*(0.5*(0.5*predict1 + 0.5*predict2) + 0.5*predict3) + 0.5*predict5
prob = 0.5*prob1 + 0.5*prob2


"""预测结果输出"""
result = pd.DataFrame({"id": data_valid_id, "prob": prob})
result.to_csv("../output/"+"result_time"+str(int(time.time()))+".txt", index=False)




























