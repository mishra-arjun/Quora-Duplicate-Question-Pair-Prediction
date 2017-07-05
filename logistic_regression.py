df = pd.read_csv('f124.csv')
question1_vectors = cPickle.load(open("q1_w2v.pkl", "rb"))
question2_vectors = cPickle.load(open("q2_w2v.pkl", "rb"))
tfidf_feat1 = cPickle.load(open("tfidf1.pkl", "rb"))
tfidf_feat2 = cPickle.load(open("tfidf2.pkl", "rb"))
tfidf_feat3 = cPickle.load(open("tfidf3.pkl", "rb"))

import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import _pickle as cPickle

idx = list(range(404290))
idx_train,idx_test = train_test_split(idx,test_size = 0.2,random_state = 300)

df = df.fillna(0)
df2 = df.copy()
data_mat = df2.as_matrix()

X_train = data_mat[idx_train,6:16]
Y_train = data_mat[idx_train,5].astype(int)

X_test = data_mat[idx_test,6:16]
Y_test = data_mat[idx_test,5].astype(int)

LR_model1 = LogisticRegression()

LR_model1.fit(X_train,Y_train)

pred_test1 = LR_model1.predict(X_test)
cnf_mat1 = confusion_matrix(Y_test,pred_test1)
sum(np.diagonal(cnf_mat1))/len(Y_test)


# #### Model2 on basic plus fuzzy features (F1+F2)

X_train = data_mat[idx_train,6:23]
Y_train = data_mat[idx_train,5].astype(int)

X_test = data_mat[idx_test,6:23]
Y_test = data_mat[idx_test,5].astype(int)


LR_model2 = LogisticRegression()


LR_model2.fit(X_train,Y_train)


pred_test2 = LR_model2.predict(X_test)

cnf_mat2 = confusion_matrix(Y_test,pred_test2)
sum(np.diagonal(cnf_mat2))/len(Y_test)


# #### Model3 on basic plus fuzzy features plus wor2vec distance features (F1+F2+F4)

X_train = data_mat[idx_train,6:34]
Y_train = data_mat[idx_train,5].astype(int)

X_test = data_mat[idx_test,6:34]
Y_test = data_mat[idx_test,5].astype(int)

LR_model3 = LogisticRegression()

LR_model3.fit(X_train,Y_train)
pred_test3 = LR_model3.predict(X_test)


cnf_mat3 = confusion_matrix(Y_test,pred_test3)
sum(np.diagonal(cnf_mat3))/len(Y_test)


# #### Model4 on wor2vec features (F5)


data_mat2 = np.hstack((question1_vectors,question2_vectors))
X_train = data_mat2[idx_train,:]
Y_train = Y_train

X_test = data_mat2[idx_test,:]
Y_test = Y_test

LR_model4 = LogisticRegression()

LR_model4.fit(X_train,Y_train)

pred_test4 = LR_model4.predict(X_test)

cnf_mat4 = confusion_matrix(Y_test,pred_test4)
sum(np.diagonal(cnf_mat4))/len(Y_test)


# #### Model5 on (F1+F2+F3+F4+F5)
data5 = np.hstack((data_mat[:,6:34],data_mat2))

X_train = data5[idx_train,:]
Y_train = Y_train

X_test = data5[idx_test,:]
Y_test = Y_test

LR_model5 = LogisticRegression()

LR_model5.fit(X_train,Y_train)

pred_test5 = LR_model5.predict(X_test)

cnf_mat5 = confusion_matrix(Y_test,pred_test5)
sum(np.diagonal(cnf_mat5))/len(Y_test)


# #### Model 6 using F3-1


X_train = tfidf_feat1[idx_train,:]
Y_train = data_mat[idx_train,5].astype(int)

X_test = tfidf_feat1[idx_test,:]
Y_test = data_mat[idx_test,5].astype(int)


LR_model6 = LogisticRegression()


LR_model6.fit(X_train,Y_train)

pred_test6 = LR_model6.predict(X_test)


cnf_mat6 = confusion_matrix(Y_test,pred_test6)
sum(np.diagonal(cnf_mat6))/len(Y_test)


# #### Model 7 using F3-2

X_train = tfidf_feat2[idx_train,:]
Y_train = data_mat[idx_train,5].astype(int)

X_test = tfidf_feat2[idx_test,:]
Y_test = data_mat[idx_test,5].astype(int)


LR_model7 = LogisticRegression()

LR_model7.fit(X_train,Y_train)

pred_test7 = LR_model7.predict(X_test)


cnf_mat7 = confusion_matrix(Y_test,pred_test7)
sum(np.diagonal(cnf_mat7))/len(Y_test)


# #### Model 8 using F3-3
X_train = tfidf_feat3[idx_train,:]
Y_train = data_mat[idx_train,5].astype(int)

X_test = tfidf_feat3[idx_test,:]
Y_test = data_mat[idx_test,5].astype(int)

LR_model8 = LogisticRegression()

LR_model8.fit(X_train,Y_train)
pred_test8 = LR_model8.predict(X_test)


cnf_mat8 = confusion_matrix(Y_test,pred_test8)
sum(np.diagonal(cnf_mat8))/len(Y_test)