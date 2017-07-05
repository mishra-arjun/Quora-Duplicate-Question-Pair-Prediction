import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import pandas as pd
import _pickle as cPickle
from sklearn.model_selection import train_test_split

#GENERAL METHODOLOGY TO PERFORM GRID SEARCHING
# train = pd.read_csv('C:\\Users\\Vyom\\Desktop\\MS Analytics\\6740\\project\\f1.csv')
# X = train.ix[:,1:]
# Y = train.ix[:,0:1].copy()
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

# #define target and predictor variables
# target = 'is_duplicate'
# predictors = Xtrain.columns

# #function that will help to perform grid search
# def modelfit(alg, Xtrain, Ytrain, Xtest, Ytest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(Xtrain[predictors].values, label=Ytrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], 
#                           nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
#         alg.set_params(n_estimators=cvresult.shape[0])
    
#     alg.fit(Xtrain[predictors], Ytrain[target], eval_metric='auc')
    
#     Xtrain_predictions = alg.predict(Xtrain[predictors])
#     Xtest_predictions = alg.predict(Xtest[predictors])
    
#     Xtrain_predprob = alg.predict_proba(Xtrain[predictors])[:,1]
#     Xtest_predprob = alg.predict_proba(Xtest[predictors])[:,1]
    
#     print ("\n Model Report")
#     print ("Accuracy : %.4g" % metrics.accuracy_score(Ytest[target].values, Xtest_predictions))
#     print ("AUC Score (Test): %f" % metrics.roc_auc_score(Ytest[target], Xtest_predprob))

# # Step 1

# #defining the model
# xgb1 = XGBClassifier(
# 	learning_rate = 0.1,
# 	n_estimators = 500,
# 	max_depth = 5,
# 	min_child_weight = 1,
# 	gamma = 0,
# 	subsample = 0.8,
# 	colsample_bytree = 0.8,
# 	objective = 'binary:logistic',
# 	scale_pos_weight = 0.5,
# 	seed=50)

# #fitting the standard model defined above
# modelfit(xgb1, Xtrain, Ytrain, Xtest, Ytest, predictors)

# #Step 2
# param_test1 = {
# 'max_depth': list(range(3,10,2)),
# 'min_child_weight': list(range(1,6,2))
# }

# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate = 0.1, n_estimators = 200, max_depth = 5,
# 	min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, silent=False,
# 	objective = 'binary:logistic', nthread = 8, scale_pos_weight = 1, seed = 27),
# 	param_grid = param_test1, scoring = 'roc_auc', n_jobs = 8, iid = False, cv = 5)
 
# gsearch1.fit(Xtrain[predictors],Ytrain[target])

# gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

# #Step 3
# param_test3 = {
# 'gamma': list([i/10.0 for i in range(0,5)])
# }

# gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=4,
# 	min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, silent = False,
# 	objective= 'binary:logistic', nthread=8, scale_pos_weight=1,seed=27),
# 	param_grid = param_test3, scoring='roc_auc',n_jobs=8,iid=False, cv=5)

# gsearch3.fit(Xtrain[predictors],Ytrain[target])

# gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

# #Step 4
# param_test4 = {
# 'subsample': list([i/10.0 for i in range(6,10)]),
# 'colsample_bytree': list([i/10.0 for i in range(6,10)])
# }

# gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=4,
# 	min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, silent=False,
# 	objective= 'binary:logistic', nthread=8, scale_pos_weight=1,seed=27), 
# 	param_grid = param_test4, scoring='roc_auc',n_jobs=8,iid=False, cv=5)

# gsearch4.fit(Xtrain[predictors],Ytrain[target])

# gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# #Step 4B
# param_test4b = {
# 'subsample': list([i/100.0 for i in range(75,90,5)]),
# 'colsample_bytree': list([i/100.0 for i in range(75,90,5)])
# }

# gsearch4b = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=4,
# 	min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, silent=False,
# 	objective= 'binary:logistic', nthread=8, scale_pos_weight=1,seed=27), 
# 	param_grid = param_test5, scoring='roc_auc',n_jobs=8,iid=False, cv=5)

# gsearch4b.fit(Xtrain[predictors],Ytrain[target])

# #Step 5
# param_test5 = {
# 'reg_alpha': list([1e-5, 1e-2, 0.1, 1, 100]),
# 'reg_lambda': list([1e-5, 1e-2, 0.1, 1, 100])
# }

# gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=200, max_depth=4,
# 	min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8, silent=False,
# 	objective= 'binary:logistic', nthread=8, scale_pos_weight=1,seed=27), 
# 	param_grid = param_test6, scoring='roc_auc',n_jobs=8,iid=False, cv=5)

# gsearch5.fit(Xtrain[predictors], Ytrain[target])

# gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# #Step 6
# xgb_final = XGBClassifier(
# 	learning_rate =0.05,
# 	n_estimators=1000,
# 	max_depth=4,
# 	min_child_weight=6,
# 	gamma=0,
# 	subsample=0.8,
# 	colsample_bytree=0.8,
# 	reg_alpha=0.005,
# 	reg_lambda=0.005,
# 	objective= 'binary:logistic',
# 	nthread=8,
# 	scale_pos_weight=0.5,
# 	seed=50)

# modelfit(xgb_final, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F1
train = pd.read_csv('f1.csv')
X = train.ix[:,1:]
Y = train.ix[:,0:1].copy()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

def modelfit(alg, Xtrain, Ytrain, Xtest, Ytest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(Xtrain[predictors].values, label=Ytrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    alg.fit(Xtrain[predictors], Ytrain[target], eval_metric='auc')
    
    Xtrain_predictions = alg.predict(Xtrain[predictors])
    Xtest_predictions = alg.predict(Xtest[predictors])
    
    Xtrain_predprob = alg.predict_proba(Xtrain[predictors])[:,1]
    Xtest_predprob = alg.predict_proba(Xtest[predictors])[:,1]
    
    print ("\n Model Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(Ytest[target].values, Xtest_predictions))
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(Ytest[target], Xtest_predprob))

#defining the model
xgb_F1 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=6,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.005,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F1, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F1 and F2
train = pd.read_csv('f12.csv')
X = train.ix[:,1:]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F2 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=6,
	min_child_weight=1,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.005,
	reg_lambda=0.001,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F2, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F1, F2, F4
train = pd.read_csv('f124.csv')
X = train.ix[:,1:]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F3 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=7,
	min_child_weight=3,
	gamma=0,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.005,
	reg_lambda=0.005,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F3, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F5
q1v = cPickle.load(open("q1_w2v.pkl", "rb"))
q2v = cPickle.load(open("q2_w2v.pkl", "rb"))
X = pd.concat([pd.DataFrame(q1v), pd.DataFrame(q2v)], axis=1, ignore_index=True)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F4 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=5,
	min_child_weight=6,
	gamma=0.3,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.01,
	reg_lambda=0.05,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F4, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F1, F2, F4, and F5
train = pd.read_csv('f124.csv')
train = pd.concat([train, pd.DataFrame(q1v), pd.DataFrame(q2v)], axis=1, ignore_index=True)
X = train.ix[:,1:]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F5 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=5,
	min_child_weight=6,
	gamma=0.3,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.01,
	reg_lambda=0.05,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F5, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F3-1
tfidf1 = cPickle.load(open("tfidf1.pkl", "rb"))
X = pd.DataFrame(tfidf1)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F6 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=7,
	min_child_weight=6,
	gamma=0.2,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.01,
	reg_lambda=0.1,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F6, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F3-2
tfidf2 = cPickle.load(open("tfidf2.pkl", "rb"))
X = pd.DataFrame(tfidf2)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F7 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=7,
	min_child_weight=6,
	gamma=0.2,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.1,
	reg_lambda=0.1,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F7, Xtrain, Ytrain, Xtest, Ytest, predictors)

# model for F3-3
tfidf3 = cPickle.load(open("tfidf3.pkl", "rb"))
X = pd.DataFrame(tfidf3)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25)

#define target and predictor variables
target = 'is_duplicate'
predictors = Xtrain.columns

#defining the model
xgb_F8 = XGBClassifier(
	learning_rate =0.05,
	n_estimators=1000,
	max_depth=7,
	min_child_weight=6,
	gamma=0.2,
	subsample=0.8,
	colsample_bytree=0.8,
	reg_alpha=0.1,
	reg_lambda=0.1,
	objective= 'binary:logistic',
	nthread=8,
	scale_pos_weight=0.2,
	seed=50)

#fitting
modelfit(xgb_F8, Xtrain, Ytrain, Xtest, Ytest, predictors)