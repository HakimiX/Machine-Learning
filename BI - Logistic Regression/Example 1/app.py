from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model, metrics, svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import ShuffleSplit
from math import sqrt
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import sklearn
import datetime
import numpy as np
import pandas as pd
import nltk # needed for Naive-Bayes


# Preprocessing 
dataframe = pd.read_json('./data/users.json')

# Determining if any value is missing
# https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/
# Chaining '.values.any()'
if (dataframe.isnull().values.any()):
    dataframe = dataframe[dataframe['error'] != 'Permission denied']
    dataframe = dataframe.dropna(subset=['created', 'karma', 'submitted'], how='all')
    dataframe = dataframe.drop(['about','error','id'], axis=1)


dataframe['created'] = dataframe['created'].astype(int).values.reshape(-1,1)
dataframe['karma'] = dataframe['karma'].astype(int).values.reshape(-1,1)
dataframe['submitted'] = dataframe['submitted'].astype(int).values.reshape(-1,1)

# Split dataset into train and test subsets (SciKit)
x_train, x_test, y_train, y_test, z_train, z_test = sklearn.model_selection.train_test_split(
    dataframe['created'].values,
    dataframe['karma'].values,
    dataframe['submitted'].values,
    test_size=0.20, random_state=5)

# Linear Regression - a linear approach for modeling the relationship between a scalar
# dependent variable 'y' and one or more explanatory variables denoted 'x'
linear_regr = linear_model.LinearRegression()

# Training
x_and_z_train = np.stack([x_train, z_train], axis=1).reshape(-1,2)
x_and_z_test = np.stack([x_test, z_test], axis=1).reshape(-1,2)

linear_regr.fit(x_and_z_train,y_train)

predict_regr = linear_regr.predict(x_and_z_test)


def kfold_cross_validation():
    # We split out data into k different subsets
    kf = KFold(n_splits=10)

    mean_absolute_error_list = []
    root_mean_square_error_list = []

    for train_index, test_index in kf.split(dataframe['created'].values, dataframe['karma'].values, dataframe['submitted'].values):
        x_and_z_temp = np.stack([dataframe['created'].values, dataframe['submitted'].values], axis=1).reshape(-1,2)
        x_and_z_train, x_and_z_test = x_and_z_temp[train_index], x_and_z_temp[test_index]
        y_train, y_test = dataframe['karma'].values[train_index], dataframe['karma'].values[test_index]

        linear_model_temp = linear_model.LinearRegression()

        linear_model_temp.fit(x_and_z_train, y_train)

        predict_regr = linear_model_temp.predict(x_and_z_test)

        mean_absolute_error_temp = str(metrics.mean_absolute_error(y_test, predict_regr))
        root_mean_square_error_temp = str(sqrt(metrics.mean_squared_error(y_test, predict_regr)))

        mean_absolute_error_list.append(mean_absolute_error_temp)
        root_mean_square_error_list.append(mean_absolute_error_temp)

        print("Mean Absolute Error (MAE): ", mean_absolute_error_temp)
        print("Root Mean Square Error (RMSE)", root_mean_square_error_temp)
        print("")
    
    mean_absolute_error_average = np.array(mean_absolute_error_list).astype(np.float)
    root_mean_square_error_average = np.array(root_mean_square_error_list).astype(np.float)

    print("Mean Absolute Error (MAE) Average, ", str(np.mean(mean_absolute_error_average)))
    print("Root Mean Square Error (RMSE) Average, ", str(np.mean(root_mean_square_error_average)))


def logistic_regression():

    dataframe_cancer = pd.read_csv('./data/breast_cancer.csv', sep=',')

    # Report the head of the table
    print(dataframe_cancer.head(10))

    logistic_regr = linear_model.LogisticRegression()

    header_temp1 = ['Concavity1','Texture1','Symmetry1']
    header_temp2 = ['Perimeter1','Area1','Compactness1']
    header_temp3 = ['Perimeter1','Area1','Compactness1','Concavity1','Texture1','Symmetry1']

    headers = [header_temp1, header_temp2, header_temp3]

    calculated_list = []

    for vars in headers:
        x = dataframe_cancer[vars].values.reshape(-1, len(vars))
        y = dataframe_cancer['Diagnosis']

        kf = KFold(n_splits=10)

        calculated_accuracy_list = []

        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            logistic_regr.fit(x_train, y_train)

            predict_regr = cross_val_predict(logistic_regr, x_test, y_test, cv=10)

            calculated_accuracy = metrics.accuracy_score(y_test, predict_regr, normalize=True)
            
            calculated_accuracy_list.append(calculated_accuracy)

            print("Accuracy: ", calculated_accuracy)
        
        calculated_list.append(calculated_accuracy_list)

    print()
    average_size = np.array(calculated_list[1]).astype(np.float)
    average_shape = np.array(calculated_list[0]).astype(np.float)
    print("Average Size: ", str(np.mean(average_size)))
    print("Average Shape: ", str(np.mean(average_shape)))


def generate_scatterplot(dataframe):
    # 3D scatter plot, compare 3 charachteristics of data instead of two     
    # https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    model = plt.figure()
    plot = model.add_subplot(111, projection='3d')
    plot.scatter(xs=x_test, ys=y_test, zs=z_test, c='r', marker='o')
    plot.set_xlabel('created')
    plot.set_ylabel('karma')
    plot.set_zlabel('submitted')
    plot.get_figure().savefig('scatterplot.png')

 
def run():

    # Part 1 
    # Split data into 80/20 training and testing and create a new multivariate linear
    # analysis. In this model, include the number of posts as an addition to time
    #generate_scatterplot(dataframe)

    # Report the MAE and RMSE
    print("Mean Absolute Error (MAE)")
    print("Training Data: ", str(metrics.mean_absolute_error(y_train, linear_regr.predict(x_and_z_train))))
    print("Test Data: ", str(metrics.mean_absolute_error(y_test, predict_regr)))
    print("Root-Mean-Square Error (RMSE)")
    print("Training Data: ", str(sqrt(metrics.mean_squared_error(y_train, linear_regr.predict(x_and_z_train)))))
    print("Test Data: ", str(sqrt(metrics.mean_squared_error(y_test, predict_regr))))
    
    
    # Part 2
    # Create 10 training/test data pairs
    # Train model using training data from current fold
    # Test model on the test data from current fold
    # Report MAE and RMSE on test data from current fold
    # Lastly take the average of each metric for all the folds
    kfold_cross_validation()


    # Part 3
    # Load dataset and report the head
    # Using 10-fold cross-validation, train your logistic model using every
    # variable in the dataset
    # Report the accuracy
    logistic_regression()

    
run()