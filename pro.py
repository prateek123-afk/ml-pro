import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(x_test)
pca=PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
ica = FastICA(n_components=2, random_state=42)
X_train_ica = ica.fit_transform(X_train_scaled)
X_test_ica = ica.transform(X_test_scaled)
model=LogisticRegression()
model.fit(X_train_pca,y_train)
y_pred=model.predict(X_test_pca)
accuracy=accuracy_score(y_test,y_pred)
print( "Accuracy is",accuracy)