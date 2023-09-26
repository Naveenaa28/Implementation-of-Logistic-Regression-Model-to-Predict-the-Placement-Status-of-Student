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
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/ed6a0ce8-3ce4-46e3-91c5-6973e42ee1ff)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/81df9fd3-f901-421d-9c5d-7907b20b385d)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/82adc13c-cd00-4d49-8fb9-a962f3afed83)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/ef895493-e527-4823-a310-fdffe9346d90)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/fcf09806-ff26-4685-ab0b-2dfda1e0c37c)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/743490f5-4079-48ce-8012-90dace7eb26c)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/a5b3f411-3b57-467b-989a-2c22cf0bb2ab)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/33ac0405-32ea-4770-9f8e-b270aae498ed)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/b77233a4-f3fe-45cc-b726-fe22228fb652)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/3f1744c5-9aa1-41de-b203-5677819d6c47)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/dac33061-d18a-4658-80f8-56bcb9ff1674)
![image](https://github.com/Naveenaa28/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/131433133/48373c71-bf5e-4642-8fac-89752df7006f)
## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
