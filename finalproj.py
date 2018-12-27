# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:27:08 2018

@author: polsa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.metrics import confusion_matrix
def senstivity(Y_test,fraud_pred):
    TN=confusion_matrix(Y_test, fraud_pred)[0][0]
    FP=confusion_matrix(Y_test, fraud_pred)[0][1]
    FN=confusion_matrix(Y_test, fraud_pred)[1][0]
    TP=confusion_matrix(Y_test, fraud_pred)[1][1]
    
    sensitivity = TP/(TP+FN)
    print ("Sensitivity",sensitivity)
    
    accuracy = (TP+TN)/(TP+FN+FP+TN)

    print("Accuracy:",accuracy)
    return accuracy
score={}
    
tr=pd.read_csv(r"C:\Users\polsa.LAPTOP-FV8LMBSP\Downloads\train_auto.csv")
print("TRAINING DATA")
    
    #PREPROCESSING
print("PREPROCESSING TRAIN DATA")
print(tr.shape)
c=['INDEX','TARGET_AMT','TARGET_FLAG']
for col in tr.columns:
    if col in c:
        del tr[col]
tr.replace('z_SUV','SUV')
#TRY AND USE KNN
def num_missing(x):
    return sum(x.isnull())
    
print ("Missing values per column:")
print (tr.apply(num_missing, axis=0)) 

tr=tr.replace(('z_SUV','z_F','z_No','z_High School','z_Blue Collar','z_Highly Rural/ Rural'),('SUV','F','No','High School','Blue Collar','Highly Rural/ Rural'))

def replace_most_common(x):
    if pd.isnull(x):
        return most_common
    else:
        return x
        
most_common=tr['YOJ'].median()
tr['YOJ'] = tr['YOJ'].map(replace_most_common)
tr['INCOME']=tr['INCOME'].replace('[\$,]', '', regex=True).astype(float)
    

most_common=tr['INCOME'].median()
tr['INCOME'] = tr['INCOME'].map(replace_most_common)

tr['HOME_VAL']=tr['HOME_VAL'].replace('[\$,]', '', regex=True).astype(float)
most_common=tr['HOME_VAL'].median()
tr['HOME_VAL'] = tr['HOME_VAL'].map(replace_most_common)

#a3=tr['AGE'].value_counts()
most_common=tr['AGE'].median()
tr['AGE'] = tr['AGE'].map(replace_most_common)


most_common=tr['CAR_AGE'].median()
tr['CAR_AGE'] = tr['CAR_AGE'].map(replace_most_common)

most_common='Blue Collar'
tr['JOB'] = tr['JOB'].map(replace_most_common)



print ("Missing values per column:")
print (tr.apply(num_missing, axis=0)) 

tr['OLDCLAIM']=tr['OLDCLAIM'].replace('[\$,]', '', regex=True).astype(float)
tr['BLUEBOOK']=tr['BLUEBOOK'].replace('[\$,]', '', regex=True).astype(float)

gender = {'M': 0,'F': 1} 
tr.SEX = [gender[item] for item in tr.SEX] 

Y_N={'Yes':0,'No':1}
tr.MSTATUS = [Y_N[item] for item in tr.MSTATUS]
tr.PARENT1 = [Y_N[item] for item in tr.PARENT1]
tr.REVOKED = [Y_N[item] for item in tr.REVOKED]

y_n={'yes':0,'no':1}
tr.RED_CAR = [y_n[item] for item in tr.RED_CAR]

use={'Private':0,'Commercial':1}
tr.CAR_USE = [use[item] for item in tr.CAR_USE]

edu={'High School':0,'Bachelors':1,'Masters':2,'<High School':3,'PhD':4}
tr.EDUCATION = [edu[item] for item in tr.EDUCATION]

job={'Blue Collar':0,'Clerical':1,'Professional':2,'Manager':3,'Lawyer':4,'Student':5,'Home Maker':6,'Doctor':7}
tr.JOB = [job[item] for item in tr.JOB]

typ={'SUV':0,'Minivan':1,'Pickup':2,'Sports Car':3,'Van':4,'Panel Truck':5}
tr.CAR_TYPE = [typ[item] for item in tr.CAR_TYPE]

u_r={'Highly Urban/ Urban':0,'Highly Rural/ Rural':1}
tr.URBANICITY = [u_r[item] for item in tr.URBANICITY]


#print(list(tr.columns.values))
tr = tr[['KIDSDRIV', 'AGE', 'HOMEKIDS','YOJ', 'INCOME', 'PARENT1', 'HOME_VAL', 'MSTATUS', 'SEX', 
         'EDUCATION', 'JOB', 'TRAVTIME', 'CAR_USE', 'BLUEBOOK', 'TIF', 'CAR_TYPE', 'RED_CAR',
         'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE', 'URBANICITY', 'REVOKED',]]
print(tr.columns.values)
tr.hist(figsize = (30, 30))
plt.show()
corrmat = tr.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ts=pd.read_csv(r"C:\Users\polsa.LAPTOP-FV8LMBSP\Downloads\train_auto.csv")
print("TRAINING DATA")

#PREPROCESSING
print("PREPROCESSING TEST DATA")
print(ts.shape)
c=['INDEX','TARGET_AMT','TARGET_FLAG']
for col in ts.columns:
    if col in c:
        del ts[col]
ts.replace('z_SUV','SUV')
#TRY AND USE KNN
def num_missing(x):
  return sum(x.isnull())

print ("Missing values per column:")
print (ts.apply(num_missing, axis=0)) 


ts=ts.replace(('z_SUV','z_F','z_No','z_High School','z_Blue Collar','z_Highly Rural/ Rural'),('SUV','F','No','High School','Blue Collar','Highly Rural/ Rural'))
#print(a.max())

def replace_most_common(x):
    if pd.isnull(x):
        return most_common
    else:
        return x
    
most_common=ts['YOJ'].median()
ts['YOJ'] = ts['YOJ'].map(replace_most_common)

ts['INCOME']=ts['INCOME'].replace('[\$,]', '', regex=True).astype(float)


most_common=ts['INCOME'].median()
ts['INCOME'] = ts['INCOME'].map(replace_most_common)

ts['HOME_VAL']=ts['HOME_VAL'].replace('[\$,]', '', regex=True).astype(float)
most_common=ts['HOME_VAL'].median()
ts['HOME_VAL'] = ts['HOME_VAL'].map(replace_most_common)

most_common=ts['AGE'].median()
ts['AGE'] = ts['AGE'].map(replace_most_common)


most_common=ts['CAR_AGE'].median()
ts['CAR_AGE'] = ts['CAR_AGE'].map(replace_most_common)

most_common='Blue Collar'
ts['JOB'] = ts['JOB'].map(replace_most_common)



print ("Missing values per column:")
print (ts.apply(num_missing, axis=0)) 

ts['OLDCLAIM']=ts['OLDCLAIM'].replace('[\$,]', '', regex=True).astype(float)
ts['BLUEBOOK']=ts['BLUEBOOK'].replace('[\$,]', '', regex=True).astype(float)

gender = {'M': 0,'F': 1} 
ts.SEX = [gender[item] for item in ts.SEX] 

Y_N={'Yes':0,'No':1}
ts.MSTATUS = [Y_N[item] for item in ts.MSTATUS]
ts.PARENT1 = [Y_N[item] for item in ts.PARENT1]
ts.REVOKED = [Y_N[item] for item in ts.REVOKED]

y_n={'yes':0,'no':1}
ts.RED_CAR = [y_n[item] for item in ts.RED_CAR]

use={'Private':0,'Commercial':1}
ts.CAR_USE = [use[item] for item in ts.CAR_USE]

edu={'High School':0,'Bachelors':1,'Masters':2,'<High School':3,'PhD':4}
ts.EDUCATION = [edu[item] for item in ts.EDUCATION]

job={'Blue Collar':0,'Clerical':1,'Professional':2,'Manager':3,'Lawyer':4,'Student':5,'Home Maker':6,'Doctor':7}
ts.JOB = [job[item] for item in ts.JOB]

typ={'SUV':0,'Minivan':1,'Pickup':2,'Sports Car':3,'Van':4,'Panel Truck':5}
ts.CAR_TYPE = [typ[item] for item in ts.CAR_TYPE]

u_r={'Highly Urban/ Urban':0,'Highly Rural/ Rural':1}
ts.URBANICITY = [u_r[item] for item in ts.URBANICITY]


#print(list(ts.columns.values))
ts = ts[['KIDSDRIV', 'AGE', 'HOMEKIDS','YOJ', 'INCOME', 'PARENT1', 'HOME_VAL', 'MSTATUS', 'SEX', 
         'EDUCATION', 'JOB', 'TRAVTIME', 'CAR_USE', 'BLUEBOOK', 'TIF', 'CAR_TYPE', 'RED_CAR',
         'OLDCLAIM', 'CLM_FREQ', 'MVR_PTS', 'CAR_AGE', 'URBANICITY', 'REVOKED',]]
print(ts.columns.values)
ts.hist(figsize = (30, 30))
plt.show()
corrmat = ts.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

ar = tr.values
X_train = ar[:,0:22]
Y_train= ar[:,22]
ar1=ar[:,20:22]

ar = ts.values
X_test = ar[:,0:22]
Y_test= ar[:,22]
ar2=ar[:,20:22]

#SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#RANDOM FOREST
print ("APPLYING RANDOM FOREST")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 15,random_state = 0)
rf.fit(X_train, Y_train)

y_rf = rf.predict(X_test)

# CONFUSION MATRIX

print("CONFUSION MATRIX RANDOM FOREST","\n",confusion_matrix(Y_test, y_rf))

acc=senstivity(Y_test, y_rf)

score['Random Forest']=acc


#k_nn
print("APPLYING k_nn")
from sklearn.neighbors import KNeighborsClassifier
k_nn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p = 2)
k_nn.fit(X_train, Y_train)

y_k_nn = k_nn.predict(X_test)

# CONFUSION MATRIX
print("KNN CONFUSION MATRIX","\n",confusion_matrix(Y_test, y_k_nn))

acc=senstivity(Y_test, y_k_nn)
score['KNN']=acc

#anomaly detection

frames = [tr,ts]
data_frame = pd.concat(frames)

ar = data_frame.values
x = ar[:,0:22]
y = ar[:,22]
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

print("ANOMALY DETECTION")
#lof

print("Applying LOF")
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Determine Number Of Fraud Cases In DataSet


c=tr['REVOKED'].value_counts()
Fraud=c[0]
Valid=c[1]

outlier_fraction = Fraud/Valid
print("Fraction of Fraud cases:",outlier_fraction)
print("Count in training data")
print('Fraud Cases: ',Fraud)
print('Valid Cases: ',Valid)

print("Applying Isolation Forest")

IF=IsolationForest(max_samples = len(x_train),random_state = 1)
IF.fit(x_test)
y_pred =IF.predict(x_test)
y_pred[y_pred == 1] = 1
y_pred[y_pred == -1] = 0

acc=senstivity(y_test, y_pred)
score['Isolation Forest']=acc
print("CONFUSION MATRIX OF ISOLATION FOREST","\n",confusion_matrix(y_test, y_pred))


print("Applying LOF")

L_O_F=LocalOutlierFactor(n_neighbors = 20)
y_pred = L_O_F.fit_predict(x_test)
y_pred[y_pred == 1] = 1
y_pred[y_pred == -1] = 0

acc=senstivity(y_test, y_pred)
score['LOF']=acc
print("CONFUSION MATRIX","\n",confusion_matrix(y_test, y_pred))


#SVM
print("APPLYING SVM")
from sklearn import svm

o_c = svm.OneClassSVM(kernel='linear', gamma=0.001, nu=0.5)

o_c.fit(x_train)

fraud_pred = o_c.predict(x_test)
unique, counts = np.unique(fraud_pred, return_counts=True)

print("SVM COUNTS")
print (np.asarray((unique, counts)).T)
fraud_pred = pd.DataFrame(fraud_pred)
fraud_pred= fraud_pred.rename(columns={0: 'prediction'})

TP = FN = FP = TN = 0
for j in range(len(y_test)):
    if y_test[j]== 0 and fraud_pred['prediction'][j] == 1:
        TP = TP+1
    elif y_test[j]== 0 and fraud_pred['prediction'][j] == -1:
        FN = FN+1
    elif y_test[j]== 1 and fraud_pred['prediction'][j] == 1:
        FP = FP+1
    else:
        TN = TN +1
score["SVM"]=accuracy_score(y_test,fraud_pred)
print ("Accuracy:",accuracy_score(y_test,fraud_pred))
sensitivity = TP/(TP+FN)
print ("Sensitivity",sensitivity)

x=score.keys()
y=score.values()
plt.plot(x,y)



