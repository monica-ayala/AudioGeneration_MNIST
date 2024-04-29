import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile as sf

def normalize_spectrogram(spectrogram, min_val, max_val):
    cleaned_spectrogram = np.nan_to_num(spectrogram, nan=0.0)
    normalized_spectrogram = (cleaned_spectrogram - min_val) / (max_val - min_val)
    return normalized_spectrogram

# def reshape(spectrogram):
#     reshaped = np.zeros((256, 64, 1))
#     reshaped[:min(spectrogram.shape[0], 256), :min(spectrogram.shape[1], 64), 0] = spectrogram[:256, :64]
#     return reshaped

def is_padding_necessary(signal, num_expected_samples):
    if len(signal) < num_expected_samples:
        return True
    return False

def right_padding(audio, num_missing_samples):
    padded_audio = np.pad(audio, (0, num_missing_samples), mode="constant")
    return padded_audio
    
def padding(audio, num_expected_samples):
    num_missing_samples = num_expected_samples - len(audio)
    padded_audio = right_padding(audio, num_missing_samples)
    return padded_audio

def save_full_stft(file_path, n_fft=512, hop_length=256, sr=22050, duration=0.74, mono=True):
    audio = librosa.load(file_path, sr=sr, duration=duration, mono=mono)[0]
    num_expected_samples = int(22050 * 0.74)
    if is_padding_necessary(audio, num_expected_samples):
        audio = padding(audio, num_expected_samples)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)[:-1]
    spectrogram = np.abs(stft)
    stft_dB = librosa.amplitude_to_db(spectrogram)
    return stft_dB

def process_all_files(directory, output_directory):
    min_max_dict = {}  
    
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            file_name = os.path.splitext(filename)[0]

            spectrogram = save_full_stft(file_path)
            min_val = np.min(spectrogram)
            max_val = np.max(spectrogram)

            min_max_dict[file_name] = {'min': min_val, 'max': max_val}

            normalized_spectrogram = normalize_spectrogram(spectrogram, min_val, max_val)

            if not np.isnan(normalized_spectrogram).any():
                spectrogram_file_name = f"{file_name}.npy"
                np.save(os.path.join(output_directory, spectrogram_file_name), normalized_spectrogram)

    np.save('min_max_values.npy', min_max_dict)

output_directory = 'spectrograms'
directory = 'C:\\Users\\mayal\\AudioGeneration_MNIST\\recordings'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

process_all_files(directory, output_directory)
