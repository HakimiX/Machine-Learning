from sklearn import linear_model, metrics
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn import datasets
import sklearn
import scipy.stats as stats
import numpy as np
import pandas as pd


# Part 1 - Accuracy and Recall
def logistic_regression():
    
    breast_cancer_dataframe = pd.read_csv('./data/breast_cancer.csv', sep=',')
    
    # Showing main classification metrics
    logistic_regr = linear_model.LogisticRegression()

    header_temp1 = ['Concavity1','Texture1','Symmetry1']
    header_temp2 = ['Perimeter1','Area1','Compactness1']
    header_temp3 = ['Perimeter1','Area1','Compactness1','Concavity1','Texture1','Symmetry1']

    headers = [header_temp1, header_temp2, header_temp3]

    calculated_list = []

    for vars in headers:
        x = breast_cancer_dataframe[vars].values.reshape(-1, len(vars))
        y = breast_cancer_dataframe['Diagnosis']

        kf = KFold(n_splits=10)

        calculated_accuracy_list = []

        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            logistic_regr.fit(x_train, y_train)

            predict_regr = cross_val_predict(logistic_regr, x_test, y_test, cv=10)

            calculated_accuracy = metrics.accuracy_score(y_test, predict_regr, normalize=True)
            
            calculated_classification = metrics.classification_report(y_test, predict_regr, target_names=['Malignant Cancer', 'Benign Cancer'])

            calculated_accuracy_list.append(calculated_accuracy)
            calculated_accuracy_list.append(calculated_classification)

            print("Classification Report")
            print("Accuracy: ", calculated_accuracy)
            print("Classification: ", calculated_classification)

        calculated_list.append(calculated_accuracy_list)


# Part 2 - Population and T-test
def population_test():

    brain_size_dataframe = pd.read_csv('./data/brain_size.csv', delimiter=';')

    height_from_dataframe = brain_size_dataframe['Height'].values.astype(float)

    height_from_dataframe_female = brain_size_dataframe[brain_size_dataframe['Gender'] == 'Female']['Height']
    height_from_dataframe_male = brain_size_dataframe[brain_size_dataframe['Gender'] == 'Male']['Height']

    groupby_gender = brain_size_dataframe.groupby('Gender')
    
    for gender, value in groupby_gender['Height']:
        print("Dataframe Average Height:", gender, value.mean())

    # Average female height in Denmark - 168.7 cm (66.41 inches)
    # Average female height in USA - 162 cm (63.77 inches)
    # Source - http://www.averageheight.co/average-female-height-by-country

    # Average male height in Denmark - 182.6 cm (71.88 inches)
    # Average male height in USA - 175.2 (69.2 inches)
    # Source - http://www.averageheight.co/average-male-height-by-country

    print()

    print("Dataframe average female height 65.76 inches compared to Denmark average female height 66.44 inches")
    t_test, population_mean = stats.ttest_1samp(height_from_dataframe_female, 66)
    print("T-test:", str(t_test), " Popmean: ", str(population_mean))
    print()
    print("Dataframe average female height 65.76 inches compared to USA average female height 65.77 inches")
    t_test, population_mean = stats.ttest_1samp(height_from_dataframe_female, 65)
    print("T-test:", str(t_test), " Popmean: ", str(population_mean))
    print()
    print("Dataframe average male height 71 inches compared to Denmark average male height 71.88 inches")
    t_test, population_mean = stats.ttest_1samp(height_from_dataframe_male, 72)
    print("T-test:", str(t_test), " Popmean: ", str(population_mean))
    print()
    print("Dataframe average male height 71 inches compared to USA average male height 69.2 inches")
    t_test, population_mean = stats.ttest_1samp(height_from_dataframe_male, 69)
    print("T-test:", str(t_test), " Popmean: ", str(population_mean))
    print()
    print("Dataframe male/female height compared to Denmark")
    t_test, population_mean = stats.ttest_1samp(height_from_dataframe, 71)
    print("T-test:", str(t_test), " Popmean: ", str(population_mean))
    print()
    print("Dataframe male/female height compared to USA")
    t_test, population_mean = stats.ttest_1samp(height_from_dataframe, 68)
    print("T-test:", str(t_test), " Popmean: ", str(population_mean))


def run():

    # Part 2
    logistic_regression()

    # Part 2
    population_test()


run()

