# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:57:31 2022

@author: venkatesh
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

df = pd.read_csv("PROJECT2.csv")
# shuffle the dataset! 
df = df.sample(frac=1).reset_index(drop=True)
df.head(10)

# split into X and Y
X = df.iloc[:,4:8]
Y = df.iloc[:,8:9].values
print(X.shape)
print(Y.shape)

# convert to numpy arrays
X = np.array(X)

# work with labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

print(encoded_Y)

# dividing X, y into train and test data
X_train, X_test, encoded_Y_train, encoded_Y_test = train_test_split(X, encoded_Y, random_state = 0)
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, encoded_Y_train)

# accuracy on X_test
accuracy = knn.score(X_test, encoded_Y_test)
print(accuracy)

knn_predictions = knn.predict(X_test)
cm = confusion_matrix(encoded_Y_test, knn_predictions)
print(cm)
data=[[70,67,45,230]]
new_df=pd.DataFrame(data,columns=['WDG temp','OIL TEMP','ambient TMP','AVG Current'])
new_df
knn = knn.predict(new_df)
print(knn)

#pickel
import pickle
filename='model.sav'
pickle.dump(knn,open(filename,'wb'))
load_model=pickle.load(open(filename,"rb"))
r=load_model.score(X_test,encoded_Y_test)
print(r)