import sklearn
import numpy as np
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%matplotlib inline 


#Cambio de dimensión 31 a 2
pca=PCA(n_components=2)
pca.fit(datos)
t=pca.transform(datos)
