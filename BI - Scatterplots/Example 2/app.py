import json
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pylab import polyfit, poly1d
from scipy.stats.stats import pearsonr

traindata = 'training.json'
testdata = 'testing.json'
  
def getData():
    global training_data
    global testing_data
    with open(traindata) as json_data_1:
        training_data = json.load(json_data_1, encoding = 'ISO-8859-1')
    with open(testdata) as json_data_2:
        testing_data = json.load(json_data_2, encoding = 'ISO-8859-1')

def formatData():
    global train_karma
    global train_created
    global test_karma
    global test_created
    global TRAIN_CREATED
    global TEST_CREATED
    
    train_karma = []
    train_created = []
    test_karma = []
    test_created = []
    TRAIN_CREATED = []
    TEST_CREATED = []
    
    for i in training_data:
        if not i.has_key('karma') or not i.has_key('created'):
            i["karma"] = 0
            i["created"] = 1509813038         
        train_karma.append(i["karma"])
        train_created.append(i["created"])

    for j in testing_data:
        if not j.has_key('karma') or not j.has_key('created'):
            j["karma"] = 0
            j["created"] = 1509813038
        test_karma.append(j["karma"])
        test_created.append(j["created"])
        
    TRAIN_CREATED = np.array([train_created])
    TRAIN_CREATED = TRAIN_CREATED.T
    TEST_CREATED = np.array([test_created])
    TEST_CREATED = TEST_CREATED.T

def trainAndPlot():
    global model
    
    X,y = TRAIN_CREATED, train_karma
    x = train_created
    
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    fit = np.polyfit(x, y, deg=1)
    fit_fn = np.poly1d(fit)
    plt.plot(X, y, 'ro', X, fit_fn(X), 'b')
    plt.show()
    #plt.savefig('HackerNewsPlot.png')
    print("a: ", model.coef_)
    print("b: ", model.intercept_)

def calcMAE():
    train_karma_pred = model.predict(TRAIN_CREATED)
    test_karma_pred = model.predict(TEST_CREATED)
    
    train_MAE = mean_absolute_error(train_karma, train_karma_pred)
    test_MAE = mean_absolute_error(test_karma, test_karma_pred)
    
    print('Train MAE: ', train_MAE)
    print('Test MAE: ', test_MAE)

def calcMSE():
    train_karma_pred = model.predict(TRAIN_CREATED)
    test_karma_pred = model.predict(TEST_CREATED)

    train_MSE = mean_squared_error(train_karma, train_karma_pred)
    test_MSE = mean_squared_error(test_karma, test_karma_pred)
    train_MSE = math.sqrt(train_MSE)
    test_MSE = math.sqrt(test_MSE)
    
    print('Train MSE: ', train_MSE)
    print('Test MSE: ', test_MSE)

def calcPR():
    train_PR = pearsonr(train_created, train_karma)
    test_PR = pearsonr(test_created, test_karma)

    print('Train PR: ', train_PR)
    print('Test PR: ', test_PR)

def run():
    getData()
    formatData()
    trainAndPlot()
    calcMAE()
    calcMSE()
    calcPR()
    print 'Score: ', model.score(, test_karma)
    print('done')

run()