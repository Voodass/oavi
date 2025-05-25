import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import wiener, savgol_filter
import soundfile as sf
from scipy.ndimage import maximum_filter

SRC_DIR     = 'audio_src'
RESULTS_DIR = 'audio_results'
os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

AUDIO_FILE    = os.path.join(SRC_DIR, "record.wav")
FILTERED_WAV_WIENER = os.path.join(RESULTS_DIR, "denoised_wiener.wav")
FILTERED_WAV_SAVGOL = os.path.join(RESULTS_DIR, "denoised_savgol.wav")
PLOT_BEFORE   = os.path.join(RESULTS_DIR, "spec_before.png")
PLOT_WIENER   = os.path.join(RESULTS_DIR, "spec_wiener.png")
PLOT_SAVGOL   = os.path.join(RESULTS_DIR, "spec_savgol.png")
PEAKS_FILE_WIENER = os.path.join(RESULTS_DIR, "peaks_wiener.txt")
PEAKS_FILE_SAVGOL = os.path.join(RESULTS_DIR, "peaks_savgol.txt")


def load_audio(file_path):
   
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

def wiener_filter(y, noise_power=0.1):
  
    fft_signal = np.fft.fft(y)
    power_signal = np.abs(fft_signal)**2
    estimated_signal = power_signal / (power_signal + noise_power)
    filtered_fft = fft_signal * estimated_signal
    return np.real(np.fft.ifft(filtered_fft))


def savgol_filter(y, window_length=51, polyorder=3):
 
    if window_length % 2 == 0:
        window_length += 1
    
    half_window = window_length // 2
    y_padded = np.pad(y, (half_window, half_window), mode='edge')
    filtered = np.zeros_like(y)
    
  
    x = np.arange(-half_window, half_window + 1)
    X = np.vander(x, polyorder + 1, increasing=True)
    
  
    X_pinv = np.linalg.pinv(X)
    
    for i in range(len(y)):
        window = y_padded[i:i + window_length]
        coeffs = np.dot(X_pinv, window)
        filtered[i] = coeffs[0] 
        
    return filtered


def plot_spectrogram(y, sr, title, out_path, peaks=None):
 
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='magma')

    if peaks:
      
        for peak in peaks:
            t, f, _ = peak
            plt.plot(t, f, 'ro') 

    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] Спектрограмма сохранена: {out_path}")


def find_time_freq_peaks(y, sr, n_fft=1024, hop_length=256, dt=0.1, df=50):
  
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    t_win = int(dt * sr / hop_length)
    f_res = freqs[1] - freqs[0]
    f_win = max(int(df / f_res), 1)

    local_max = maximum_filter(S, size=(f_win, t_win))
    peaks = np.argwhere(S == local_max)

    peaks_list = [(times[t], freqs[f], S[f, t]) for f, t in peaks]
    peaks_list.sort(key=lambda x: x[2], reverse=True)
    return peaks_list[:10]


def save_peaks_to_file(peaks, file_path):
   
    with open(file_path, 'w') as file:
        for peak in peaks:
            t, freq, amplitude = peak  
            file.write(f"Time: {t:.3f}s, Frequency: {freq:.2f}Hz, Amplitude: {amplitude:.2f}\n")
    print(f"[✓] Пики сохранены в файл: {file_path}")



def main():
   
    y, sr = load_audio(AUDIO_FILE)
    print(f"[i] Загружено: {AUDIO_FILE}, sr={sr}, длительность={len(y)/sr:.2f}s")

  
    plot_spectrogram(y, sr, 'Original', PLOT_BEFORE)


    y_wiener = wiener_filter(y)
    sf.write(FILTERED_WAV_WIENER, y_wiener, sr)
    print(f"[✓] Фильтр Винера сохранён: {FILTERED_WAV_WIENER}")
    
    
    peaks_wiener = find_time_freq_peaks(y_wiener, sr)
  
    save_peaks_to_file(peaks_wiener, PEAKS_FILE_WIENER)
  
    plot_spectrogram(y_wiener, sr, 'Wiener Denoised', PLOT_WIENER, peaks_wiener)

  
    y_savgol = savgol_filter(y)
    sf.write(FILTERED_WAV_SAVGOL, y_savgol, sr)
    print(f"[✓] Фильтр Савицкого-Голея сохранён: {FILTERED_WAV_SAVGOL}")

 
    peaks_savgol = find_time_freq_peaks(y_savgol, sr)
   
    save_peaks_to_file(peaks_savgol, PEAKS_FILE_SAVGOL)
    
    plot_spectrogram(y_savgol, sr, 'Savitzky-Golay Denoised', PLOT_SAVGOL, peaks_savgol)

if __name__ == "__main__":
    main()
