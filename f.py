%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy import misc
from scipy.fftpack import fft, ifft

#Almacena la imagen Arbol.png
im= misc.imread('Arbol.png')

#FFT de la imagen
x=np.linspace(0,10,256)
y = fft(im)
plt.plot(x,y)
plt.legend()
plt.ylabel('T. de Fourier')
plt.savefig('SalazarNicole_FT2D.pdf')

from scipy import fftpack
im_fft = fftpack.fft2(im)

# Show the results

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

plt.figure()
plot_spectrum(im_fft)
plt.title('Fourier transform')


keep_fraction = 0.1
im_fft2 = im_fft.copy()
r, c = im_fft2.shape
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plot_spectrum(im_fft2)
plt.title('Filtered Spectrum')


im_new = fftpack.ifft2(im_fft2).real

plt.figure()
plt.imshow(im_new, plt.cm.gray)
plt.title('Reconstructed Image')
