# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:25:30 2022

@author: venkatesh
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import re
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
    return render_template('home.html', query="")



@app.route("/", methods=['POST'])
def faultPrediction():
    
    df = pd.read_csv("PROJECT2.csv")
    df = df.sample(frac=1).reset_index(drop=True)
    df.info()

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    
    # split into X and Y
    X = df.iloc[:,4:8]
    Y = df.iloc[:,8:9].values

    # convert to numpy arrays
    X = np.array(X)

    # work with labels
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

# dividing X, y into train and test data
    X_train, X_test, encoded_Y_train, encoded_Y_test = train_test_split(X, encoded_Y, random_state = 0)
# training a KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, encoded_Y_train)
# accuracy on X_test
    accuracy = knn.score(X_test, encoded_Y_test)

    data=[['winding temperature','oil temperature','ambient temperature','Average current']]
    new_df=pd.DataFrame(data,columns=['WDG temp','OIL TEMP','ambient TMP','AVG Current'])
    new_df
    knn = knn.predict(new_df)
    
    
    if knn==0:
            output1 = "NO FAULT"   
    elif knn==1:
            output1 = "HEATING FAULT"
    elif knn==2:
            output1 = "OVER HEATING"
    elif knn==3:
            output1 = "OVER LOADING"       
    else:
            output1= "SHORTCIRCUIT"

    return render_template('home.html',output=output1,query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'])
    
app.run()