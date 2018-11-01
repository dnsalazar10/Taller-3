import sklearn
import numpy as np
import mglearn
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
%matplotlib inline 

cancer=load_breast_cancer()
#Carga archivo WDBC.dat
datos=np.loadtxt('WDBC1.dat', delimiter=',')

#Cambio de dimensión 31 a 2
pca=PCA(n_components=2)
pca.fit(datos)
t=pca.transform(datos)

#Imprime matriz de covarianza
def cov_matrix(data):
    n_dim = np.shape(data)[1]
    n_points = np.shape(data)[0]
    cov = np.ones([n_dim, n_dim])
    for i in range(n_dim):
        for j in range(n_dim):
            mean_i = np.mean(data[:,i])
            mean_j = np.mean(data[:,j])
            cov[i,j] = np.sum((data[:,i]-mean_i) * (data[:,j]-mean_j)) / (n_points -1)
    return cov

#Covarianza
cov = cov_matrix(t)
print(cov)




