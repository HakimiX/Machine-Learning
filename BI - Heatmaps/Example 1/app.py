import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import nltk
import math
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import folium
from folium.plugins import HeatMap

# Part 1
def vader_lexicon():

    hackernews_items = pd.read_csv('./data/hn_items.csv',delimiter=',',encoding='latin-1')

    nltk.download('vader_lexicon')

    model = SentimentIntensityAnalyzer()

    hn_text = hackernews_items.dropna(subset=['text'], how='all')['text'].values

    df = pd.DataFrame(columns=['text','neg','pos'])

    data = []

    for text in hn_text:
        score = model.polarity_scores(text)
        
        data.append({'text':text,'neg':score['neg'],'pos':score['pos']})
        

    df = df.append(data)

    print('5 most positive: ')
    print(df.nlargest(5,'pos'))

    print('5 most negative: ')
    print(df.nlargest(5,'neg'))

    # Part 2
    # Kfold Cross validation

    negatives = df['neg']
    positives = df['pos']

    folds = KFold(n_splits=10)
    kmeans = KMeans(n_clusters=6)
    classifier = KNeighborsClassifier()

    X = negatives.values
    Y = positives.values
    XY = np.stack((X,Y),axis=1)

    for train_idx, test_idx in folds.split(XY):
        idx = np.concatenate([train_idx,test_idx])
        XY_fold = XY[idx]
        
        kmeans.fit(XY_fold)
        
        labels = kmeans.labels_
        
        classifier.fit(XY_fold,labels)
        
        print(metrics.accuracy_score(labels, classifier.predict(XY_fold)))

# Part 3
def price_proximity():

    boliga = pd.read_csv('./data/boliga_zealand.csv').drop(['Index', '_1', 'Unnamed: 0'], axis=1)

    zip_df = pd.DataFrame(boliga['zip_code'].str.split(' ',1).tolist(), columns = ['zip','city'])

    boliga = boliga.assign(zip_int=zip_df['zip'])
    boliga['zip_int'] = pd.to_numeric(boliga['zip_int'], errors='coerce')
    boliga = boliga[boliga['zip_int'] <= 2999]

    heatmap_df = boliga[['lon','lat','price']].dropna()

    boliga_map = folium.Map(location=[55.676098, 12.568337], zoom_start=11)

    folium.Marker(location=[55.676098, 12.568337], icon=folium.Icon(color='red',icon='home')).add_to(boliga_map)

    heat_data = [(e.lat,e.lon,float(e.price)) for e in heatmap_df.itertuples()]

    HeatMap(heat_data, radius=7).add_to(boliga_map)

    boliga_map.save('heatmap.html')


def run():

    #vader_lexicon()
    price_proximity()

run()