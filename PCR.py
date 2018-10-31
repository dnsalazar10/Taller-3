import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg 
import sklearn
import mglearn
from sklearn.decomposition import PCA
from scipy.fftpack import fft, ifft, fftfreq
from mpl_toolkits.mplot3d import Axes3D

#### Punto 2: PCA y PCR ####
#Carga de datos
datos = np.loadtxt('WDBC.dat',delimiter=',')

#Funcion de matriz de covarianza 
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

#Cambio de dimensión 31 a 2
pca=PCA(n_components=2)
pca.fit(datos)
t=pca.transform(datos)

#Covarianza
cov = cov_matrix(t)
print(cov)

#Vectores y valores propios
val, vec = numpy.linalg.eig(cov)

a1=val[0]
a2=val[1]
b1=vec[:,0]
b2=vec[:,1]

#Imprime a qué autovalor corresponde cada autovector
print(f'Autovector: {b1} y Autovalor: {a1}')
print(f'Autovector: {b2} y Autovalor: {a2}')

#Proyección en PC1 y PC2
mglearn.discrete_scatter(t[:,0],t[:,1])
plt.legend(loc='best')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
#plt.savefig(SalazarNicole_PCA.pdf)
