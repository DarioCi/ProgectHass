import numpy as np
import SigOp.segmenter as sg
import matplotlib.pyplot as plt
import SigOp.common as cm

class Bcg:

  def __init__(self, path):
    self.path = path

  def __str__(self):
    return f"{self.path}"

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

s1= Bcg('C:/Users/dario/Desktop/Hassisto/Prova2/sleeprawlive/sleeprawlive.csv')
print("il path di elaborazione è: " + s1.path)
s1.elabora(5)
