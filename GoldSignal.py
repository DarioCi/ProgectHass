import numpy as np
import SigOp.segmenter as sg
import matplotlib.pyplot as plt
import SigOp.common as cm
from scipy import signal

class Gold:
  def __init__(self, path):
    self.path = path

  def __str__(self):
    return f"{self.path}"

  def elabora(self):
    data = np.genfromtxt(self.path, delimiter=';')
    plt.plot(data[15000:15800], label="Segnale di partenza gold")
    plt.legend()
    plt.figure()
    goldresample = signal.resample(data[15000:15400], 800)
    plt.plot(goldresample, label="Gold da 100hz -> 50hz")
    plt.legend()
    goldnormalize = sg.renormalize_signal(data[15000:15800], 100)
    goldresample2 = signal.resample(goldnormalize[:400], 800)
    plt.plot(goldresample2, label="Gold a 50hz normalizzato")
    plt.legend()
    plt.show()

    s1=Gold('C:/Users/dario/Desktop/Hassisto/Prova2/waveform-gold/gold.csv')
    print("il path di elaborazione è: " + s1.path)
    s1.elabora()