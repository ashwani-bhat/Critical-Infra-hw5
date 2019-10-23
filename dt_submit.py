#Importing all necessary libraries 
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
import sys

# func used for fitting and prediction score for validation set 
def get_model(model,X_train,y_train,X_val,y_val):
    clf = model.fit(X_train, y_train)
    predict = clf.predict(X_val)
    score = clf.score(X_val,y_val)
    return (clf,score)
	
#Reading Data from csv file
data = pd.read_csv('HW_TESLA.csv')

# Separating Labels from data
X = data.iloc[:,1:].values
y = data.iloc[:,:1].values

# Splitting Data into Train and Test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
print(X_train.shape)
# Using KFold with K=3
kf = KFold(n_splits=3) 

# Storing score for each validation set
val_score= list()

for train_index, test_index in kf.split(X_train):
    #print("TRAIN:", train_index, 'TEST:', test_index)
    X_tr, X_val = X_train[train_index], X_train[test_index]
    y_tr, y_val = y_train[train_index], y_train[test_index]
    clf,score = get_model(DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=12, min_samples_split=2, min_samples_leaf=2)
                          ,X_tr,y_tr,X_val,y_val)
    val_score.append(score)
    #print('Score: ',score)

# Calculating Mean validation score     
print("Mean Validation Score: ",np.mean(val_score))

#Predict labels for test data set
predict = clf.predict(X_test)

#Representing FP,FN,TP,TN in Matrix
cm = confusion_matrix(y_test,predict)
#z = [x for x in predict if x==0]


cr = classification_report(y_test,predict)
tn= cm[0][0]
fp= cm[0][1]
fn= cm[1][0]
tp= cm[1][1]

# Calculating Accuracy, Precision,Recall
accuracy = (tp+tn)/(tp+tn+fp+fn)
accuracy = accuracy*100
precision = (tp)/(tp+fp)
recall= (tp)/(tp+fn)

print("\n-------------------For our Test Dataset-------------------")
print("\nAccuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)

print("\n Confusion Matrix::")
print(cm)
print("\n  Classification Report::")
print(cr)


# Testing for the provided test dataset in test.csv file

test = pd.read_csv('test.csv')
r,w = test.shape
if r==0:
	sys.exit()
X_t = test.iloc[:,1:].values
y_t = test.iloc[:,:1].values

#Predict labels for test data
predict = clf.predict(X_t)

#Representing FP,FN,TP,TN in Matrix
cm = confusion_matrix(y_t,predict)
#z = [x for x in predict if x==0]
#print(len(z))
print(cm)
cr = classification_report(y_t,predict)
tn= cm[0][0]
fp= cm[0][1]
fn= cm[1][0]
tp= cm[1][1]

# Calculating Accuracy, Precision,Recall
accuracy = (tp+tn)/(tp+tn+fp+fn)
accuracy = accuracy*100
precision = (tp)/(tp+fp)
recall= (tp)/(tp+fn)

print("\n-------------------For your Test Dataset----------------")
print("\nAccuracy: ",accuracy)
print("Precision: ",precision)
print("Recall: ",recall)

print("\n Confusion Matrix::")
print(cm)
print("\n Classification Report::")
print(cr)