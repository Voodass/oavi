import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

SRC_DIR = 'src'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_spectrogram_with_formants(y, sr, formants, title, outpath, formant_color='red'):
    """Спектрограмма с формантами (логарифмическая шкала)"""
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512,
                                 x_axis='time', y_axis='log', 
                                 cmap='viridis', ax=ax)
    
    for i, f in enumerate(formants, start=1):
        if not np.isnan(f):
            ax.axhline(y=f, color=formant_color, linestyle='--', 
                      linewidth=1.5, alpha=0.7,
                      label=f'Formant {i}: {f:.1f} Hz' if i == 1 else "")
    
    plt.colorbar(img, format='%+2.0f dB')
    plt.title(title)
    ax.set_ylim(50, sr//2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_f0_contour(y, sr, title, outpath):
    f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr,
                    frame_length=2048, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    plt.figure(figsize=(8,4))
    plt.plot(times, f0, label='F0 contour')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_spectral_peaks(y, sr, harmonics, formants, title, outpath):
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec_avg = D.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_db = librosa.amplitude_to_db(spec_avg, ref=np.max)
    plt.figure(figsize=(8,4))
    plt.plot(freqs, spec_db, label='Average spectrum (dB)')

    for k, h in enumerate(harmonics, start=1):
        plt.axvline(x=h, linestyle='--', label=f'Harmonic {k}: {h:.1f} Hz')

    for i, f in enumerate(formants, start=1):
        if not np.isnan(f):
            plt.axvline(x=f, color='red', linestyle=':', label=f'Formant {i}: {f:.1f} Hz')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def min_max_frequency(y, sr, threshold_db=-60):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mean_spec = S_db.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = mean_spec > threshold_db
    if not mask.any():
        return 0.0, 0.0
    return freqs[mask].min(), freqs[mask].max()

def estimate_f0(y, sr, frame_length=2048):
    """Улучшенная оценка F0 с медианной фильтрацией"""
    f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr,
                    frame_length=frame_length, hop_length=512)
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        return np.nan
    return np.median(valid_f0)

def estimate_harmonics(y, sr, f0, n_harmonics=10):
    """Оценка гармоник на основе F0"""
    if np.isnan(f0) or f0 <= 0:
        return []
    
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_avg = D.mean(axis=1)
    
    harmonics = []
    for k in range(1, n_harmonics+1):
        h_freq = f0 * k
        if h_freq >= sr/2:
            break
        idx = np.argmin(np.abs(freqs - h_freq))
        if spec_avg[idx] > 0.3 * spec_avg.max():  # Более мягкий порог
            harmonics.append(h_freq)
    return harmonics

def estimate_formants_v2(y, sr, n_formants=3, max_bandwidth=500):
    """Улучшенная оценка формант"""
    # Автоматический выбор порядка LPC
    lpc_order = 2 * n_formants + 2
    
    # Выбор сегмента с максимальной энергией вместо центрального
    frame_length = int(0.03 * sr)  # 30 мс
    if frame_length > len(y):
        frame_length = len(y)
    
    energy = np.convolve(y**2, np.ones(frame_length), mode='valid')
    start = np.argmax(energy)
    y_segment = y[start:start+frame_length]
    
    # Нормализация и оконная функция
    y_segment = librosa.util.normalize(y_segment)
    window = np.hamming(len(y_segment))
    y_windowed = y_segment * window
    
    # LPC с проверкой на устойчивость
    try:
        a = librosa.lpc(y_windowed, order=lpc_order)
        roots = np.roots(a)
        roots = roots[(np.imag(roots) > 0) & (np.abs(roots) < 1.0)]  # Комплексные корни внутри единичной окружности
        
        # Преобразование корней в частоты и полосы
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * sr / (2 * np.pi)
        bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))
        
        # Фильтрация по полосе и частоте
        mask = (bandwidths < max_bandwidth) & (freqs > 50) & (freqs < 5000)
        freqs = freqs[mask]
        
        # Дополнительная фильтрация: удаление дубликатов (ближе 50 Гц)
        if len(freqs) > 1:
            unique_freqs = [freqs[0]]
            for f in freqs[1:]:
                if np.min(np.abs(np.array(unique_freqs) - f)) > 50:
                    unique_freqs.append(f)
            freqs = np.array(unique_freqs)
        
        if len(freqs) >= n_formants:
            return np.sort(freqs)[:n_formants]
        
    except Exception as e:
        print(f"LPC error: {str(e)}")
    
    # Если что-то пошло не так, возвращаем NaN вместо дефолтных значений
    return np.array([np.nan] * n_formants)

def analyze_audio_file(path):
    """Полный анализ аудиофайла"""
    name = os.path.splitext(os.path.basename(path))[0]
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # Основные характеристики
    fmin, fmax = min_max_frequency(y, sr)
    f0 = estimate_f0(y, sr)
    harmonics = estimate_harmonics(y, sr, f0)
    formants = estimate_formants_v2(y, sr)
    
    # Визуализация
    plot_spectrogram_with_formants(y, sr, formants,
                                 f'Spectrogram with Formants: {name}',
                                 os.path.join(RESULTS_DIR, f'spec_formants_{name}.png'))
    
    plot_f0_contour(y, sr, f'F0 Contour: {name}',
                   os.path.join(RESULTS_DIR, f'f0_{name}.png'))
    
    plot_spectral_peaks(y, sr, harmonics, formants,
                       f'Spectral Peaks: {name}',
                       os.path.join(RESULTS_DIR, f'peaks_{name}.png'))
    
    return {
        'name': name,
        'fmin': fmin,
        'fmax': fmax,
        'f0': f0,
        'overtones': harmonics,
        'formants': [f if not np.isnan(f) else None for f in formants]
    }

def main():
    files = glob.glob(os.path.join(SRC_DIR, '*.wav'))
    report = []
    
    for path in files:
        print(f"Processing {os.path.basename(path)}...")
        try:
            result = analyze_audio_file(path)
            report.append(result)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    # Сохранение отчета
    report_path = os.path.join(RESULTS_DIR, 'report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            for item in report:
                f.write(f"\nFile: {item['name']}\n")
                f.write(f"Safe name: {item['safe_name']}\n")
                f.write(f"Frequency range: {item['fmin']:.1f} - {item['fmax']:.1f} Hz\n")
                f.write(f"Fundamental frequency: {item['f0']:.1f} Hz\n")
                f.write(f"Harmonics: {', '.join(f'{h:.1f}' for h in item['overtones'])}\n")
                
                formants_str = []
                for i, f in enumerate(item['formants'], 1):
                    if f is not None:
                        formants_str.append(f"F{i}: {f:.1f} Hz")
                    else:
                        formants_str.append(f"F{i}: Not detected")
                
                f.write("Formants: " + ", ".join(formants_str) + "\n")
                f.write("Generated files: " + ", ".join(item['output_files']) + "\n")
        
        print(f"\nAnalysis complete. Results saved to:")
        print(f"- Report: {report_path}")
        print(f"- Total files processed: {len(report)}/{len(files)}")
        
    except Exception as e:
        print(f"Error saving report: {str(e)}")

if __name__ == '__main__':
    main()
