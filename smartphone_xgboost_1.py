#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 7 10:00:33 2017

@author: mohit
"""

'''
We will use XGboost to predict the activity based on smartphone data. The details of the data set can be found on the following 
website - https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones/home
'''

#import libraires
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, auc


#Define the random seed
np.random.seed(24)


#import data

train = pd.read_csv('train.csv')
validationData = pd.read_csv('test.csv')

#data exploration

#look for null values
print(train.isnull().sum())
print(validationData.isnull().sum())
#no null values found

#shuffle the dataset

train = train.sample(frac=1).reset_index(drop=True)

#Extract features
X_features = train.iloc[:, :561]

#explore features
X_features.head(1)


#look at various statistics of the features columns columns

X_stats = X_features.describe()
print(X_stats)

#all values are normalized between -1 and 1


#get the labels
Y_labels = train.iloc[:, 562]

#see the distribution of various activities

labels_count = Y_labels.value_counts()
labels_count.plot(kind="bar")
print(Y_labels.value_counts())

#convert labels to Categorical data 

Y_labels_cat, _ = pd.factorize(Y_labels)

#
#model xgboost from SKLearn Library
xgb_model = xgb.XGBClassifier({'tree_method': 'gpu_hist', 'num_class':6})

#set parameetrs to explore
parameters = {
              'objective':['multi:softprob'],
              
              'learning_rate': [0.01, 0.05], #so called `eta` value
              'max_depth': [5],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10, 100], #number of trees, change it to 1000 for better results
              'seed': [24],
              'gamma': [1/10.0 for i in range(0, 3)],
              'n_jobs': [-1]
              
              }

#Grid search
#
grid_search = GridSearchCV(estimator = xgb_model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5, verbose=True)
#
grid_search = grid_search.fit(X_features, Y_labels_cat)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_



#use xgboost API now and use best parameters

#convert data into XGBoost DMatrix

XY_Dmatrix = xgb.DMatrix(X_features, label=Y_labels_cat)

#define the parameters for XGBoost

param = {
            'objective': 'multi:softprob',
            'num_class': 6,
            'learning_rate': 0.05, #eta value
            'max_depth': 5,
            'n_estimators': 100, #number of trees
            'seed': 24,
            'n_jobs':-1,
            'min_child_weight': 11,
            'silent': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'gamma': 0.1,
        }

#run XGboost cross-validation
cv_xgb = xgb.cv(params = param, dtrain = XY_Dmatrix, num_boost_round = 3000, nfold = 5,
                metrics = ['mlogloss'],
                early_stopping_rounds = 100, verbose_eval=True)

cv_xgb.tail(5)

#pick the last round number from tail and run the final model

final_gb = xgb.train(param, XY_Dmatrix, num_boost_round = 3000)

final_gb.save_model('smartphone_xgboost')

#explore important features

xgb.plot_importance(final_gb, max_num_features=20)


#predict the test results

validationData_X = validationData.iloc[:, :561]
validationData_Y = validationData.iloc[:, 562]

validationData_test = xgb.DMatrix(validationData_X)

y_pred = final_gb.predict(validationData_test)

#binarize the test data lables to use in ROC and other performance metrices

from sklearn.preprocessing import label_binarize
#the sequence is important to keep the classes same as in training data
y_test = label_binarize(validationData_Y, classes=['LAYING','WALKING','SITTING','WALKING_DOWNSTAIRS', 'STANDING', 'WALKING_UPSTAIRS'] )


#Plot ROC Curve for each activity

falsePositiveRate = dict()
truePositiveRate = dict()
rocAucScore = dict()
castDict = {0:'LAYING',1:'WALKING', 2:'SITTING', 3:'WALKING_DOWNSTAIRS', 4:'STANDING', 5:'WALKING_UPSTAIRS'}

for i in range(y_test.shape[1]):
   falsePositiveRate[i], truePositiveRate[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
   rocAucScore[i] = auc(falsePositiveRate[i], truePositiveRate[i])
    
    
#ROC curve for each activity
    
for i in range(y_test.shape[1]):
    plt.figure(i)
    plt.plot(falsePositiveRate[i], truePositiveRate[i], color='green',
             lw=1, linestyle='-.', label='ROC curve (area = %0.2f) for %s' % (rocAucScore[i], castDict[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for the activity')
    plt.legend(loc="upper left")
    plt.show()
    
#confusion matrix

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))


#F1 Score, Recall and Precision
print(classification_report(y_test, y_pred, target_names=['LAYING','WALKING','SITTING','WALKING_DOWNSTAIRS', 'STANDING', 'WALKING_UPSTAIRS']))