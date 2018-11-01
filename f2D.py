import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft2


####Transformada de Fourier####
#Carga de datos signal.dat
data1=np.loadtxt('signal.dat', delimiter=',')
#Ejes X y Y
x=data1[:,0]
y=data1[:,1]

#Carga de cadot incompleto.dat
data2=np.loadtxt('incompletos.dat', delimiter=',')
#Ejes X1 y Y1
x1=data2[:,0]
y1=data2[:,1]


#Gráfica de los datos
plt.plot(x,y, label='signal')
plt.legend( )
plt.savefig('SalazarNicole_signal.pdf')

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

#Fecuencias principales
spectrum = np.fft.fft(data1)
feq=np.fft.fftfreq(len(spectrum))

def temp_h(A):
    for l in range(0,len(A)):   
        if (A[l] > -0.03 and A[l] < 0.03):        
            for recorrido in range (1,len(A)):
                for posicion in range(len(A)-recorrido):
                    if A[posicion]>A[posicion+1]:
                        temp=A[posicion]
                        A[posicion] = A[posicion+1]
                        A[posicion+1]=temp
            print ('las frecuencias principales de la señal son:',A[l])

temp_h(feq)

#Filtroque pasa bajos de frecuencia Fc=1000Hz
Fc=np.copy (a)

for i in range(len(a)):
    if(abs(freq[i])>1000):
        Fc[i]=0
        
F=np.fft.fft(y)
plt.plot(x,F)





