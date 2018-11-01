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

#Imprime los autovalores y autovectores
val, vec = np.linalg.eig(cov)

#Vectores y valores propios
val, vec = np.linalg.eig(cov)

a1=val[0]
a2=val[1]
b1=vec[:,0]
b2=vec[:,1]

#Imprime a qué autovalor corresponde cada autovector
print(f'Autovector: {b1} y Autovalor: {a1}')
print(f'Autovector: {b2} y Autovalor: {a2}')

#Proyección de datos a PC1 y PC2
mglearn.discrete_scatter(t[:,0],t[:,1], cancer.target)
plt.legend(cancer.target_names,loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('SalazarNicole_PCA.pdf')

#Mensaje sobre utilidad del método
print("es bastante util este m´etodo ya que logra poner graficamente informacion con varios parametros y separarla en la informacion m´as importante como lo es el diagnostico")

