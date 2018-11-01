import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

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


