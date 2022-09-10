# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:43:18 2022

@author: 28629
"""

import pandas as pd
import sklearn.svm as svm
import sklearn.tree as tree
from sklearn.feature_selection import SelectKBest, chi2,f_classif
import sklearn.model_selection as ms

def input_data(path):#读取数据并划分xy
    data =  pd.read_csv(path)
    data_y = data.copy()
    del data['label']
    del data['subjectName']
    drop_colums = list(data_y.columns)
    drop_colums = drop_colums[:len(drop_colums)-1]
    data_y = data_y.drop(drop_colums,axis = 1)
    return data,data_y

def svm_train_test(X,y,X_test,y_test):
    
    clf_svc = svm.SVC()
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001,'scale','auto'],}
    gs_svm = ms.GridSearchCV(clf_svc, param_grid,verbose=1)
    gs_svm.fit(X,y.values.ravel())
    print(gs_svm.best_score_)
    print(gs_svm.score(X_test,y_test))
    print(gs_svm.best_params_)
    return 

def decision_train_test(X,y,X_train,y_test):
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree.fit(X, y)
    print(clf_tree.score(X_train, y_test))
    return 


if __name__ == "__main__":
    
    """
    读取三个csv中的数据
    
    """
    df_trainFeature1,df_trainFeature1_y = input_data("./trainData/trainFeature1.csv")
    df_trainHistFeature2,df_trainHistFeature2_y = input_data("./trainData/trainHistFeature2.csv")
    df_trainHogFeature3,df_trainHistFeature3_y = input_data("./trainData/trainHogFeature3.csv")
    
    df_train_x = pd.concat([df_trainFeature1,df_trainHistFeature2,df_trainHogFeature3],axis=1)
    df_train_y = df_trainFeature1_y
    
    
    df_testFeature1,df_testFeature1_y = input_data("./testData/testFeature1.csv")
    df_testHistFeature2,df_testHistFeature2_y = input_data("./testData/testHistFeature2.csv")
    df_testHogFeature3,df_testHogFeature3_y = input_data("./testData/testHogFeature3.csv")
    
    df_test_x = pd.concat([df_testFeature1,df_testHistFeature2,df_testHogFeature3],axis=1)
    df_test_y = df_testFeature1_y;
    
    select_model = SelectKBest(f_classif,k=20)
    
    df_train_x_new = select_model.fit_transform(df_train_x,df_train_y.values.ravel())
    
    features = list(select_model.get_feature_names_out())
    
    df_test_x_new = df_test_x[features]
    
    svm_train_test(df_train_x, df_train_y,df_test_x, df_test_y)

    """
    clf_svc = svm.SVC()
    clf_svc.fit(df_train_x_new,df_train_y.values.ravel())
    print(clf_svc.score(df_test_x_new,df_test_y.values.ravel()))
    print(clf_svc.kernel)
    print(clf_svc.gamma)
    """
    
    
    """
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree.fit(df_trainFeature1, df_trainFeature1_y)
    print(clf_tree.score(df_testFeature1, df_testFeature1_y))
    """