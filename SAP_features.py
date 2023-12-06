# Create a deployable version of all SAP features
from re import S
import numpy as np
import librosa
import scipy.signal as signal

from scipy.fft import fft, ifft, fftfreq, fftshift

# Create an object that will store all of the song features
class SAP_features:
    # Create a constructor for the object
    def __init__(self, y, fs, window_size=1024, hop_length=128, mel_spec=False):
        # Song variables
        self.y = y.astype(float)
        self.fs = fs

        # Variables associated with features
        self.window_size = window_size
        self.hop_length = hop_length

        # Calculate all features
        t, f, Sxx, gof, fm, am, sd, ent, amp = calculate_features(self.y, self.window_size, self.hop_length, fs)

        # Create a sonogram
        self.t = t
        self.f = f 
        self.sonogram = Sxx
        self.spectral_derivative = sd

        # Feature Derivatives 
        self.goodness_of_fit = gof

        # Amplitude
        self.amplitude = amp

        # Pitch Estimation Measures RAW
        self.peak_freq = self.f[np.argmax(self.sonogram, axis=0)]
        self.mean_freq = self.f.dot(self.sonogram**2) / np.sum(self.sonogram**2, axis=0)

        # Pitch Estimation Measures using YIN algorithm
        self.yin_freq = librosa.yin(y=self.y, fmin=50, fmax=2000, sr=self.fs, win_length=self.window_size, hop_length=self.hop_length)

        # Wiener Entropy
        self.entropy = ent

        # Modulation
        self.frequency_modulation = fm
        self.amplitude_modulation = am
        
# Helper function
# Actually calculate all features to use in SAP structure
def calculate_features(x, window_size, hop_length, fs, num_tapers=2):
    tapers = signal.windows.dpss(window_size, 1.5, 2)
    size = len(x)
    f_notShifted = fftfreq(window_size, 1/fs)
    f = fftshift(f_notShifted)
    f_index = f > 0

    sonogram = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))
    freq_deriv = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))
    time_deriv = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))

    goodness_of_fit = np.zeros(np.floor(size / hop_length).astype(int))
    frequency_modulation = np.zeros(np.floor(size / hop_length).astype(int))
    amplitude_modulation = np.zeros(np.floor(size / hop_length).astype(int))
    spectral_derivative = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))
    entropy = np.zeros(np.floor(size / hop_length).astype(int))
    amplitude = np.zeros(np.floor(size / hop_length).astype(int))

    wav_smp = np.arange(size-window_size, step=hop_length).astype(int)
    t = np.arange(np.floor(size / hop_length)) *(hop_length/fs)

    for i in range(len(wav_smp)):
        samps = np.arange(wav_smp[i], np.floor(wav_smp[i] + window_size).astype(int))
        window1 = x[samps] * tapers[0]
        window2 = x[samps] * tapers[1]

        # If the window has values, calculate the cepstrum
        if(window1.any()):
            real_cepstrum = fftshift(np.real(ifft(np.log10(fft(window1)))))
            goodness_of_fit[i] = np.max(real_cepstrum[f_index])
        else:
            goodness_of_fit[i] = 0
        
        powSpect1 = fftshift(fft(window1))
        powSpect2 = fftshift(fft(window2))

        r1 = (np.abs(powSpect1) + np.abs(powSpect2))**2
        sonogram[:,i] = r1[f_index]

        # Getting time and frequency derivatives
        fR1 = np.real(powSpect1[f_index])
        fi1 = np.imag(powSpect1[f_index])
        fR2 = np.real(powSpect2[f_index])
        fi2 = np.imag(powSpect2[f_index])

        time_deriv[:,i] = -fR1*fR2 - fi1*fi2
        freq_deriv[:,i] = fi1*fR2 - fR1*fi2

        # Getting frequnecy modulation
        frequency_modulation[i] = np.arctan((np.max(time_deriv[:,i])/np.max(freq_deriv[:,i]))+0.1)
        amplitude_modulation[i] = time_deriv[:,i].sum()

        # Solving for spectral derivatives
        cFM = np.cos(frequency_modulation[i])
        sFM = np.sin(frequency_modulation[i])
        spectral_derivative[:,i] = time_deriv[:,i].dot(cFM) + freq_deriv[:,i].dot(sFM)

        # Compute entropy
        sumLog = np.sum(np.log(sonogram[10:,i])) / (f_index.sum()-10)
        sumSon = np.sum(sonogram[10:,i]) / (f_index.sum()-10)
        
        # Same as -log(sumLog / sumSon)
        entropy[i] = (np.log(sumSon - sumLog) / np.log2(f_index.sum() - 10))-1

        # Amplitude
        amplitude[i] = np.log(sonogram[10:,i]).sum()

    return t, f[f_index], sonogram, goodness_of_f# Create a deployable version of all SAP features
from re import S
import numpy as np
import librosa
import scipy.signal as signal

from scipy.fft import fft, ifft, fftfreq, fftshift

# Create an object that will store all of the song features
class SAP_features:
    # Create a constructor for the object
    def __init__(self, y, fs, window_size=1024, hop_length=128, mel_spec=False):
        # Song variables
        self.y = y.astype(float)
        self.fs = fs

        # Variables associated with features
        self.window_size = window_size
        self.hop_length = hop_length

        # Calculate all features
        t, f, Sxx, gof, fm, am, sd, ent, amp = calculate_features(self.y, self.window_size, self.hop_length, fs)

        # Create a sonogram
        self.t = t
        self.f = f 
        self.sonogram = Sxx
        self.spectral_derivative = sd

        # Feature Derivatives 
        self.goodness_of_fit = gof

        # Amplitude
        self.amplitude = amp

        # Pitch Estimation Measures RAW
        self.peak_freq = self.f[np.argmax(self.sonogram, axis=0)]
        self.mean_freq = self.f.dot(self.sonogram**2) / np.sum(self.sonogram**2, axis=0)

        # Pitch Estimation Measures using YIN algorithm
        self.yin_freq = librosa.yin(y=self.y, fmin=50, fmax=2000, sr=self.fs, win_length=self.window_size, hop_length=self.hop_length)

        # Wiener Entropy
        self.entropy = ent

        # Modulation
        self.frequency_modulation = fm
        self.amplitude_modulation = am
        
# Helper function
# Actually calculate all features to use in SAP structure
def calculate_features(x, window_size, hop_length, fs, num_tapers=2):
    tapers = signal.windows.dpss(window_size, 1.5, 2)
    size = len(x)
    f_notShifted = fftfreq(window_size, 1/fs)
    f = fftshift(f_notShifted)
    f_index = f > 0

    sonogram = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))
    freq_deriv = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))
    time_deriv = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))

    goodness_of_fit = np.zeros(np.floor(size / hop_length).astype(int))
    frequency_modulation = np.zeros(np.floor(size / hop_length).astype(int))
    amplitude_modulation = np.zeros(np.floor(size / hop_length).astype(int))
    spectral_derivative = np.zeros((f_index.sum(), np.floor(size / hop_length).astype(int)))
    entropy = np.zeros(np.floor(size / hop_length).astype(int))
    amplitude = np.zeros(np.floor(size / hop_length).astype(int))

    wav_smp = np.arange(size-window_size, step=hop_length).astype(int)
    t = np.arange(np.floor(size / hop_length)) *(hop_length/fs)

    for i in range(len(wav_smp)):
        samps = np.arange(wav_smp[i], np.floor(wav_smp[i] + window_size).astype(int))
        window1 = x[samps] * tapers[0]
        window2 = x[samps] * tapers[1]

        # If the window has values, calculate the cepstrum
        if(window1.any()):
            real_cepstrum = fftshift(np.real(ifft(np.log10(fft(window1)))))
            goodness_of_fit[i] = np.max(real_cepstrum[f_index])
        else:
            goodness_of_fit[i] = 0
        
        powSpect1 = fftshift(fft(window1))
        powSpect2 = fftshift(fft(window2))

        r1 = (np.abs(powSpect1) + np.abs(powSpect2))**2
        sonogram[:,i] = r1[f_index]

        # Getting time and frequency derivatives
        fR1 = np.real(powSpect1[f_index])
        fi1 = np.imag(powSpect1[f_index])
        fR2 = np.real(powSpect2[f_index])
        fi2 = np.imag(powSpect2[f_index])

        time_deriv[:,i] = -fR1*fR2 - fi1*fi2
        freq_deriv[:,i] = fi1*fR2 - fR1*fi2

        # Getting frequnecy modulation
        frequency_modulation[i] = np.arctan((np.max(time_deriv[:,i])/np.max(freq_deriv[:,i]))+0.1)
        amplitude_modulation[i] = time_deriv[:,i].sum()

        # Solving for spectral derivatives
        cFM = np.cos(frequency_modulation[i])
        sFM = np.sin(frequency_modulation[i])
        spectral_derivative[:,i] = time_deriv[:,i].dot(cFM) + freq_deriv[:,i].dot(sFM)

        # Compute entropy
        sumLog = np.sum(np.log(sonogram[10:,i])) / (f_index.sum()-10)
        sumSon = np.sum(sonogram[10:,i]) / (f_index.sum()-10)
        
        # Same as -log(sumLog / sumSon)
        entropy[i] = (np.log(sumSon - sumLog) / np.log2(f_index.sum() - 10))-1

        # Amplitude
        amplitude[i] = np.log(sonogram[10:,i]).sum()

    return t, f[f_index], sonogram, goodness_of_f