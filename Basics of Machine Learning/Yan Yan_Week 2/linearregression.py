
# Importing required libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

# Reading the csv file.
df = pd.read_csv('data_hours.csv')
# Displayinng the first five elements of the dataframe.
print(df.head(10))

# Taking the Hours and Scores column of the dataframe as X and y
# respectively and coverting them to numpy arrays.
X = np.array(df['Hours']) # why the reshape will influence the w???
y = np.array(df['Scores'])


# Ploting the data X(Hours) on x-axis and y(Scores) on y-axis using plt.scatter()

### YOUR CODE GOES HERE   ###
fig = plt.figure(figsize=(8,6))
plt.scatter(X,y)

#Split X and y into 80% training set and 20% test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# Compute the closed-form solution for w and b for the training dataset, as given in the lecture 

### YOUR CODE GOES HERE   ###
w = np.sum((X_train - np.mean(X_train)) * (y_train - np.mean(y_train))) / np.sum(np.square(X_train - np.mean(X_train)))
b=np.mean(y_train)-w*np.mean(X_train)
# reg_model=LinearRegression()
# reg_model.fit(X_train,y_train)

# Show the regression line in the same plot using plt.plot(...)

### YOUR CODE GOES HERE   ###
#w=reg_model.coef_[0]
#b=reg_model.intercept_
y_predict=w*X+b
plt.plot(X,y_predict,'r')
plt.title('Hours vs Percentage')
plt.xlabel('X (Input) : Hours')
plt.ylabel('y (Target) : Scores')
plt.savefig("Scatterplot & Regression Line.png")

plt.show()
print('w=',w,'b=',b)


#compute the MSE for the training and test set
y_train_predict=w*X_train+b
y_test_predict=w*X_test+b

MSE_train =mean_squared_error(y_train,y_train_predict)

MSE_test = mean_squared_error(y_test ,y_test_predict)
 
print('MSE_train', MSE_train, 'MSE_test', MSE_test)

