from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd


# Preprocessing Dataset
dataframe = pd.read_json('./data/users.json').dropna(subset=['created','karma'], how='all')
dataframe = dataframe.drop(['about','error','id'], axis=1)

dataframe['created'] = dataframe['created'].astype(int).values.reshape(-1,1)
dataframe['karma'] = dataframe['karma'].astype(int).values.reshape(-1,1)

# Split dataset into train and test subsets (SciKit)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    dataframe['created'].values, 
    dataframe['karma'].values, 
    test_size=0.2, random_state=5) 

# Linear Regression - a linear approach for modeling the relationship between a scalar
# dependent variable 'y' and one or more explanatory variables denoted 'x'
linear_regr = linear_model.LinearRegression()

# Train the model on the training data using linear regression model
linear_regr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))
predict_regr = linear_regr.predict(x_test.reshape(-1,1))

# Coefficient - represents the mean change in the response variable for one unit 
# of change in the predictor variable while holding the predictor in the model constant
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
coefficient_variable = linear_regr.coef_
# The intercept (often labeled as the constant) is the expected mean value of 'y'
intercept_variable = linear_regr.intercept_


def generate_scatterplot(dataframe):
    plot = dataframe.plot(kind='scatter', x='created', y='karma')
    plot.scatter(x_test, y_test)
    plt.plot(x_test, predict_regr, color='black', linewidth=4)
    plot.get_figure().savefig('scatterplot.png')

def run():
    
    # Part 1 
    # Plot the training data in a scatter with linear regression

    #generate_scatterplot(dataframe)
    print("Coefficient: ", coefficient_variable)
    print("Intercept: ", intercept_variable)

    # Part 2
    # Calculate the mean absolute error (MAE) for training data and test data

    print("Mean Absolute Error (MAE)")
    print('Training data: ' + str(metrics.mean_absolute_error(y_train,linear_regr.predict(x_train.reshape(-1,1)))))
    print('Test data: ', str(metrics.mean_absolute_error(y_test, predict_regr)))

    # Part 3
    # Calculate the mean squared error (MSE) for the Training data and Test data

    print("Mean Squared Error (MSE)")
    print("Training data: ", str(metrics.mean_squared_error(y_train, linear_regr.predict(x_train.reshape(-1,1)))))
    print("Test data: ", str(metrics.mean_squared_error(y_test, predict_regr)))

    # Part 4
    # Calculate Peasons's 'r' value for the the Training data and test data

    print("Pearsons r value")
    print("Training data: ", str(metrics.r2_score(y_train, linear_regr.predict(x_train.reshape(-1,1)))))
    print("Test data: ", str(metrics.r2_score(y_test, predict_regr)))


run()