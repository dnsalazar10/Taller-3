import numpy as np
import matplotlib.pyplot as plt

####Transformada de Fourier####
data1=np.loadtxt('signal.dat', delimiter=',')
data2=np.loadtxt('incompletos.dat', delimiter=',')

#Ejes X y Y
x=data1[:,0]
y=data1[:,1]

#Gráfica de los datos
#plt.plot(x,y, label='signal')
#plt.legend( )
#plt.savefig('SalazarNicole_signal.pdf')

#Transformada discreta de Fourier
def fourier(y,k_m):
    n = len(y)
    X = np.zeros(k_m, dtype=np.complex)
    for k in range (0,k_m):
        t = np.arange(0, n)         
        X[k] = np.sum(y*np.exp(-2j*np.pi*t*(k/n))) 
    return X
a=fourier(y,512)

#Normaliza los datos 
n=data1.shape[0]
Fourier_n = a/n

#Las frecuencias
dt = 1
freq = fftfreq(n, dt) 

#Gráfica de la transformada de Fourier
plt.plot(freq,abs(Fourier_n),'orange',label='T.Fourier')
plt.legend()
plt.savefig('SalazarNicole_TF.pdf')

print('las frecuencias principales de la señal son', np.max(freq))
