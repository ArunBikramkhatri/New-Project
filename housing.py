# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 22:58:01 2021

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:56:47 2021

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Data with salary and ID
file0 = pd.read_csv("F:/ML Datasets/sample_submission.csv")

#Data with all columns
file1 = pd.read_csv("F:/ML Datasets/train.csv")

#Data missing salary
file2 = pd.read_csv("F:/ML Datasets/test.csv")

#Removing ID 
file0 = file0.drop(['Id'],axis=1)
    
test_data = pd.concat([file2,file0],axis=1) 

# making a whole dataset
data = pd.concat([file1 , test_data])
data = data.drop('Id',axis=1)  
#data =  data.drop(data.iloc[:,1],axis=1)



empty = []

for i in data.columns:
    if(data[i].isnull().sum()>2000):
        empty.append(i)
    
data = data.drop(empty,axis=1)    

#For missing values

#first creating new dataframe for numerical and categorical data
numerical_nan = pd.DataFrame()
categorical_nan = pd.DataFrame()


for i in range(len(data.columns)):
  #Separating numerical and categorical columns  
    if(type(data.iloc[1,i]) == str):
                
       categorical_nan = pd.concat([categorical_nan,data.iloc[:,i]],axis=1)
        
    elif(type(data.iloc[1,i])== np.int64 or (type(data.iloc[1,i])) == np.float64):
        numerical_nan = pd.concat([numerical_nan,data.iloc[:,i]],axis=1)


#Simple Imputer for categorical and numerical
numerical_imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
categorical_imputer = SimpleImputer(missing_values=np.nan , strategy='most_frequent')

#Replacing nan by most freuent for categorical data
for i in range(len(categorical_nan.columns)):

    categorical_nan.iloc[:,i] = categorical_imputer.fit_transform(categorical_nan.iloc[:,i].values.reshape(-1,1))

#Replacing nan by mean for numerical data
for j in range(len(numerical_nan.columns)):
    numerical_nan.iloc[:,j] = numerical_imputer.fit_transform(numerical_nan.iloc[:,j].values.reshape(-1,1))


#concating categprical and numerical data after removing nan values
full_data = pd.concat([categorical_nan, numerical_nan ],axis=1)

# x = independent variable and y is dependent varaiabl
x = full_data.iloc[:,:-1]
y = full_data.iloc[:,-1].values.reshape(-1,1)

#checking if there are values or not
#a = full_data.isnull().sum()

num_indices = []
cate_indices = []


for i in range(len(x.columns)):
    if(x.iloc[:,i].dtype == 'O'):
        cate_indices.append(i)
    elif(x.iloc[:,i].dtype == "int64" or x.iloc[:,i].dtype == "float64"):
        num_indices.append(i)
    
cate_label_indices = []
cate_onehot_indices = []

for i in cate_indices:
    if(len(x.iloc[:,i].value_counts())==2):
        cate_label_indices.append(i)
        
    elif(len(x.iloc[:,i].value_counts()) > 2):
        cate_onehot_indices.append(i)

label = LabelEncoder()
for i in cate_label_indices:
    x.iloc[:,i] = label.fit_transform(x.iloc[:,i])
    
#ct = ColumnTransformer(transformers=[('encoder') , OneHotEncoder() , cate_onehot_indices] , remainder='passthrough')

#x = ct.fit_transform(full_data)
ct = ColumnTransformer([("encoder" , OneHotEncoder() , cate_onehot_indices)] , remainder="passthrough")
x = ct.fit_transform(x)
x = x.toarray()

#Feature Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

#checking score of different models
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


lin_val_score = cross_val_score(LinearRegression() , x, y ,cv=10)
svr_score = cross_val_score(SVR() , x , y, cv = 10)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y ,test_size=0.2 ,random_state=0)

regressor = SVR()
regressor.fit(x_train , y_train)
y_pred = regressor.predict(x_test)


y_pred = y_pred.reshape(len(y_pred),1)
y_test = y_test.reshape(len(y_test),1)

concat = np.concatenate((y_pred,y_test),axis=1)
