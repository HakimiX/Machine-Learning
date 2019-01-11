# Assignment 8 - Population and Deep Learning

## Part 1 - Accuracy and Recall

### Accuracy and Precision

Accuracy and Precision are both important factors when taking data measurements. They both reflect how close a measurement is to an actual value. 

__Accuracy__ is how close a measured value is to the actual value. The closer the value is to the actual value, the more accurate the measurement. Accuracy is more important when trying to hit a target. 

__Precision__ is how close the measured values are to each other. The closer each measurement is to the other measurements, the more precise the measurement. Precision is more important in calculations. 

### Precision and Recall

Precision and Recall are both important factors when trying to understand and measure relevance. High precision means that we have more relevant results than irrelevant ones. 

__Recall__ also known as sensitivity is the ratio of correctly predicted observations or relevant instances that have been retrieved over the total amount of instances.


![Text](https://github.com/HakimiX/BusinessIntelligence/blob/master/Assignment8/Model/model.jpg)

Precision is accuracy of positive predictions and is around 0.80-0.90 and is somewhat consistent The F1 Score is a helpful metric for comparing two classifiers (benign, malignant), it takes into account precision and the recall, and is created by finding the mean of precision and recall. 

## Part 2 - Population and T-test

The `brain_size.csv` dataset contains female and male height. We were not sure whether to compare male and female together or separately, so we have done both

### Individual Gender Comparison

__Average Height__

* Average female height in Denmark - 168.7 cm (66.41 inches)

* Average female height in USA - 162 cm (63.77 inches)

* Average male height in Denmark - 182.6 cm (71.88 inches)

* Average male height in USA - 175.2 (69.2 inches)

[Source](http://www.averageheight.co/average-female-height-by-country)

__Dataframe Average Height__

```python
groupby_gender = brain_size_dataframe.groupby('Gender')
    
    for gender, value in groupby_gender['Height']:
        print("Dataframe Average Height:", gender, value.mean())
```
```python
Dataframe Average Height: Female 65.765
Dataframe Average Height: Male 71.41
```

Dataframe average female height 65.76 inches compared to Denmark average female height 66.44 inches
```python
t_test, population_mean = stats.ttest_1samp(height_from_dataframe_female, 66)
```
```python
T-test: 1.49511066685  Popmean:  0.15131013469
```

Dataframe average female height 65.76 inches compared to USA average female height 65.77 inches
```python
t_test, population_mean = stats.ttest_1samp(height_from_dataframe_female, 65)
```
```python
T-test: 1.49511066685  Popmean:  0.15131013469
```

Dataframe average male height 71 inches compared to Denmark average male height 71.88 inches
```python
t_test, population_mean = stats.ttest_1samp(height_from_dataframe_male, 72)
```
```python
T-test: -0.825318221249  Popmean:  0.419433209147
```

Dataframe average male height 71 inches compared to USA average male height 69.2 inches
```python
t_test, population_mean = stats.ttest_1samp(height_from_dataframe_male, 69)
```
```python
T-test: 3.37121510713  Popmean:  0.00320647462375
```

### Population Comparison

__Population Average Height__ 
```python
Denmark average height: 71
USA average height: 68.4
```

Dataframe both male and female height compared to Denmark (71 inches)
```python
t_test, population_mean = stats.ttest_1samp(height_from_dataframe, 71)
```
```python
T-test: -3.85063165253  Popmean:  0.00042684721728
```

Dataframe both male and female height compared to USA (68.4 inches)
```python
t_test, population_mean = stats.ttest_1samp(height_from_dataframe, 68)
```
```python
T-test: 0.937718588959  Popmean:  0.354160229104
```






