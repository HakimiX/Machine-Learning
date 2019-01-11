import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, metrics
from sklearn.model_selection import KFold, cross_val_predict
import scipy.stats as stats
import sklearn.linear_model as lm

# Part 1
def classification_cancer():

    df = pd.read_csv('./data/breast_cancer.csv', sep=',')
    #print(df.head(10))

    logreg = linear_model.LogisticRegression()

    descriptors = ['Perimeter1','Area1','Compactness1','Concavity1','Texture1','Symmetry1']

    X = df[descriptors].values.reshape(-1,len(descriptors))
    Y = df['Diagnosis']

    folds = KFold(n_splits=10)

    accuracies = []
        
    print()

    for train_idx, test_idx in folds.split(X, Y):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        logreg.fit(X_train, Y_train)

        pred = cross_val_predict(logreg,X_test,Y_test, cv=10)

        accuracy = metrics.classification_report(Y_test, pred, target_names=["Benign","Malignant"])

        accuracies.append(accuracy)

        print(accuracy)

# Part 2
def brain_cancer():

    brain_df = pd.read_csv('./data/brain_size.csv', delimiter=';', index_col=0)
    # brain_df.head(5)

    # We have rows with bad data in them, so we have to remove those to be 
    # able to use this sample properly
    brain_df = brain_df[brain_df['Height'] != '.']

    height_data = brain_df['Height'].values.astype(float)
    height_data.shape

    t, P = stats.ttest_1samp(height_data,71)

    print("T-value: " + str(t))
    print("P-value: " + str(P))

    t, P = stats.ttest_1samp(height_data,68.4)

    print("T-value: " + str(t))
    print("P-value: " + str(P))


# Part 3 - Optional
#def cross_perceptron():

#    perc = lm.Perceptron(max_iter=100, tol=None)

#    folds = KFold(n_splits=10)

#    for train_idx, test_idx in folds.split(X, Y):
#        X_train, X_test = X[train_idx], X[test_idx]
#        Y_train, Y_test = Y[train_idx], Y[test_idx]

#        perc.fit(X_train, Y_train)

#        pred = cross_val_predict(perc,X_test,Y_test, cv=10)

#        accuracy = metrics.classification_report(Y_test, pred, target_names=["Benign","Malignant"])

#        print(accuracy)
    


def run():

    #classification_cancer()
    brain_cancer()
    

run()
    