import numpy as np
import scipy
from scipy import integrate

import SigOp.segmenter as sg
import matplotlib.pyplot as plt
import SigOp.common as cm
import pandas as pd
from numpy import trapz

class Bcg:

  path = ""

  def setPath(self, elaborationPath):
    self.path = elaborationPath


  def grafdati(self,iniziof,finef):
    data = np.genfromtxt(self.path, delimiter=';')
    plt.plot(data[iniziof:finef], label="Segnale di partenza BCG")
    plt.legend()
    plt.show()
    return data[iniziof:finef]
    '''
    def elabora(self,cutoff):
    data = np.genfromtxt(self.path, delimiter=';')
    plt.plot(data[15000:15800], label="Segnale di partenza BCG")
    plt.legend()
    plt.figure()
    datanormalize = sg.renormalize_signal(data[15000:15800], 50)
    plt.plot(datanormalize, label="Segnale di partenza normalizzato a 0")
    plt.legend()
    plt.figure()
    filteredl = cm.filter_lowpass(datanormalize, 50, cutoff)
    plt.plot(filteredl, label="Segnale normalizzato filtrato a 5Hz")
    plt.legend()
    plt.figure()
    plt.show()
    return datanormalize
    '''

  def elaboration(self, inputData, cutoff):
    normalizedData = sg.renormalize_signal(inputData, 50)
    filteredData = cm.filter_lowpass(normalizedData, 50, cutoff)
    return filteredData, normalizedData

  def normalizeData(self, inputData):
    return sg.renormalize_signal(inputData, 50)

  def filterlowData (self, inputData, cutoff):
    return cm.filter_lowpass(inputData, 50, cutoff)

  def varstdminmax (self,segnale):
    datafrm=pd.DataFrame(data=segnale, columns=['BCG'])
    print(datafrm.describe())
    arraypicchi=scipy.signal.find_peaks(segnale)
    print("Posizione dei picchi =", arraypicchi)
    return arraypicchi

  def area (self,segnale):
    datafrm = pd.DataFrame(data=segnale, columns=['BCG'])
    ax = datafrm.plot.area(stacked=False, legend="BCG")
    area = trapz(segnale, dx=1)
    print("Area =", area, "m^2")

  def fourier (self,segnale):
    f = np.fft.fft(segnale - np.mean(segnale))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum2 = (np.abs(fshift))
    plt.subplot(212)
    plt.plot(magnitude_spectrum2)
    plt.title('ballist ts_spectrum')
    plt.show()

  '''
  def coeff (self, segnale,npunti):
    xs = np.linspace(0, 4 * np.pi, npunti)
    pvdata = pd.read_csv(segnale, delimiter=";")
    pvdata.columns = ['signal', 'time']
    f1 = np.sin(xs)
    f2 = 2 * np.sin(xs / 2) + 0.5 * np.sin(10 * xs)
    f3 = signal.square(xs)
    f4 = 2 * np.sin(xs / 2) + 0.5 * np.sin(10 * xs) + np.random.normal(0, 2, xs.shape)'''

  def coeff(self,li, lf, n, f):
    l = (lf - li) / 2
    # Constant term
    a0 = 1 / l * integrate.quad(lambda x: f(x), li, lf)[0]
    # Cosine coefficents
    A = np.zeros((n))
    # Sine coefficents
    B = np.zeros((n))

    for i in range(1, n + 1):
      A[i - 1] = 1 / l * integrate.quad(lambda x: f(x) * np.cos(i * np.pi * x / l), li, lf)[0]
      B[i - 1] = 1 / l * integrate.quad(lambda x: f(x) * np.sin(i * np.pi * x / l), li, lf)[0]



    return [a0 / 2.0, A, B]

  def rolling_rms(self,x, N):
    xc = np.cumsum(abs(x) ** 2);
    return np.sqrt((xc[N:] - xc[:-N]) / N)

s1= Bcg()
s1.setPath('C:/Users/dario/Desktop/Hassisto/Prova2/sleeprawlive/sleeprawlive.csv')
#print("il path di elaborazione ??: " + s1.path)
datii=s1.grafdati(15000,15800)
rmsfilter=s1.rolling_rms(datii,300)
plt.plot(rmsfilter,label="rms filter")
plt.legend()
plt.show()
normalizzato=s1.normalizeData(datii)

s1.varstdminmax(normalizzato)
'''
coeffs=s1.coeff(-np.pi,np.pi,3,datii)
print('Fourier coefficients for the function wave\n')
print('a0 ='+str(coeffs[0]))
print('an ='+str(coeffs[1]))
print('bn ='+str(coeffs[2]))
print('-----------------------\n\n')
'''

