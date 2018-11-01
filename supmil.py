import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq


###Frecuencias superiores a 1000Hz###

#Transformada discreta de Fourier
def fourier(y,k_m):
    n = len(y)
    X = np.zeros(k_m, dtype=np.complex)
    for k in range (0,k_m):
        t = np.arange(0, n)         
        X[k] = np.sum(y*np.exp(-2j*np.pi*t*(k/n))) 
    return X
a=fourier(y,512)

 Fc=np.copy (F)

for i in range(len(F)):
	if(abs(freq[i])>1000):
		Fc[i]=0


