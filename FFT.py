%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import misc
from scipy.fftpack import fft, ifft

#FFT de la imagen
x=np.linspace(0,10,256)
y = fft(im)
plt.plot(x,y)
plt.legend()
plt.ylabel('T. de Fourier')
plt.savefig('SalazarNicole_FT2D.pdf')
