# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values. 
 ## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Naveenaa V.R
RegisterNumber:212221220035  
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
print("Placement data:")
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or col
print("Salary data:")
data1.head()

print("Checking the null() function:")
data1.isnull().sum()

print ("Data Duplicate:")
data1.duplicated().sum()

print("Print data:")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

print("Data-status value of x:")
x=data1.iloc[:,:-1]
x

print("Data-status value of y:")
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print ("y_prediction array:")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") #A Library for Large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score =(TP+TN)/
#accuracy_score(y_true,y_pred,normalize=False)
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
print("Confusion array:")
confusion

from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("Classification report:")
print(classification_report1)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/09ae9e76-6d0c-4570-a6f2-81fbe1bef58e)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/52786781-0971-478b-9870-7a22ff341edd)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/9dd5622d-f3a4-44ca-9ad7-5d94022ebc1c)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/8f74dba2-cdb1-499a-bd35-be5c1e758215)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/c10ac17e-aed7-4acb-b862-a48a2cd10a59)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/0bfaaf0c-f90b-4d18-8d93-2d78f6236e82)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/c3466eeb-2f69-4514-b73f-bcafc1c60a56)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/c4651087-aa7f-4034-9747-c847c3e668f5)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
