# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe.

4.Plot the required graph both for test data and training data and Find the values of MSE , MAE and RMSE.

## Program:
```
/*Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sanjay S
RegisterNumber: 22007761
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
#segregation data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#spliting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted value
Y_pred
#displaying actual value
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(Y_test,Y_pred)
print('MSC=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:

![ml2op1](https://user-images.githubusercontent.com/115128955/200998386-d5fbc49a-e91e-46b3-88e0-1c41293b559e.png)

![ml2op2](https://user-images.githubusercontent.com/115128955/200998447-bf5450d5-6e50-4c1f-a50e-e6a83112494e.png)

![ml2op3](https://user-images.githubusercontent.com/115128955/200998469-1f943ff5-7d5d-44b0-973d-b66f38a0593c.png)

![ml2op4](https://user-images.githubusercontent.com/115128955/200998486-2a003a77-67e7-4d4d-917e-a791090f7e9e.png)

![ml2op5](https://user-images.githubusercontent.com/115128955/200998526-9138583e-b94c-447d-862c-eaa611d505ed.png)

![ml2op6](https://user-images.githubusercontent.com/115128955/200998545-13312be6-5900-43fb-bbbb-cd28b81c9faa.png)

![ml2op7](https://user-images.githubusercontent.com/115128955/200998570-8e37d3dd-9565-445b-9e93-8205f51353db.png)

![ml2op8](https://user-images.githubusercontent.com/115128955/200998614-6820432e-e367-48fc-8d84-88f3145f16b3.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
