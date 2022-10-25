import numpy as np
import SigOp.segmenter as sg
import matplotlib.pyplot as plt
import SigOp.common as cm
from scipy import signal

class Gold:

  path = ""

  def setPath(self, elaborationPath):
    self.path = elaborationPath

  def normalizeData(self, inputData,signalfrequency):
      return sg.renormalize_signal(inputData, signalfrequency)

  def resample(self,signalfrequency,windin,windout,ncampioni):
      data = np.genfromtxt(self.path, delimiter=';')
      goldnormalize = sg.renormalize_signal(data[windin:windout], signalfrequency)
      goldresample = signal.resample(goldnormalize[:(ncampioni/2)], ncampioni)
      plt.plot(goldresample,label="Segnale normalizzato e ricampionato")
      return goldresample

  def plot(self,signal):
      plt.plot(signal)
      plt.show()



s1 = Gold()
s1.setPath('C:/Users/dario/Desktop/Hassisto/Prova2/waveform-gold/gold.csv')
# print("il path di elaborazione è: " + s1.path)
s1.resample()