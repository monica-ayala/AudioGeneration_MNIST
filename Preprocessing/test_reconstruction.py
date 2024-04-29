import librosa
import soundfile as sf
import numpy as np

def inverse_normalize_spectrogram(normalized_spectrogram, min_val, max_val):
    return normalized_spectrogram * (max_val - min_val) + min_val

def reconstruct_audio(stft, output_path, min_val, max_val):
    magnitude_db = inverse_normalize_spectrogram(stft, min_val, max_val)
    magnitude_linear = librosa.db_to_amplitude(magnitude_db)
    y_reconstructed = librosa.istft(magnitude_linear, hop_length=256)
    sf.write(output_path, y_reconstructed, 22050)

spectrogram = np.load('spectrograms/0_george_0.npy')
min_max_dict = np.load('min_max_values.npy', allow_pickle=True).item()

file_name = '0_george_0'  
min_val = min_max_dict[file_name]['min']
max_val = min_max_dict[file_name]['max']

reconstruct_audio(spectrogram, 'new.wav', min_val, max_val)
