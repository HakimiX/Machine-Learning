# Assignment 5 - Linear Machine Learning

## Part 1 

Coefficient - represents the mean change in the response variable for one unit of change in the predictor variable while holding the predictor in the model constant. When the y value increases the x value decreases. 

The intercept (often labeled as the constant) is the expected mean value of "y"

```python
print("Coefficient: ", coefficient_variable)
print("Intercept: ", intercept_variable)
```

```python
generate_scatterplot(dataframe)
```

__Output__

```python
Coefficient: -4.18548395e-05
```

```python
Intercept: 61994.37153162
```

![Text](https://github.com/HakimiX/BusinessIntelligence/blob/master/Assignment5/scatterplot.png)


## Part 2

__Mean Absolute Error (MAE)__

Mean Absolute Error (MAE) difference between two continuous variables. The mean absolute error is an average of the absolute error, which means the smaller the number the better it is. The numbers we get by running the code are fairly far from 0, which makes the mean absolute model inefficient. 

```python
print('Training data: ' + str(metrics.mean_absolute_error(y_train,linear_regr.predict(x_train.reshape(-1,1)))))
print('Test data: ', str(metrics.mean_absolute_error(y_test, predict_regr)))
```

__Output__

```python
Training data: 4482.32480631
```

```python
Test data:  4366.59123274
```

## Part 3

__Mean Squared Error (MSE)__

Mean Squared Error measures the average of the squares of the errors or deviations - that is, the difference between the estimator and what is estimated. Just like MAE, the smaller the number the better it is. The mean sqaured error is also a bad fit. The difference occurs because the estimator doesn't account for information (outliers) that could produce a more accurate estimate. 

```python
print("Training data: ", str(metrics.mean_squared_error(y_train, linear_regr.predict(x_train.reshape(-1,1)))))
print("Test data: ", str(metrics.mean_squared_error(y_test, predict_regr)))
```

__Output__

```python
Training data: 100613209.759
```

```python
Test data:  78059377.8934
```

## Part 4

__Pearson's r value__

Pearson's r value is a measure of the linear correlation between two variables x and y. it has a value between +1 and -1, where 1 is total positive linear correlation, 0 is no linear correlation, and -1 is total negative linear correlation. The numbers we get by running the code are close to 0 which makes Pearson's r value the best fit. 

```python
print("Training data: ", str(metrics.r2_score(y_train, linear_regr.predict(x_train.reshape(-1,1)))))
print("Test data: ", str(metrics.r2_score(y_test, predict_regr)))
```

__Output__

```python
Training data: 0.127692903639
```

```python
Test data:  0.132688206996
```

