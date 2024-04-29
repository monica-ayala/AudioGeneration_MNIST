import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = 'untitled.wav'
audio_data, sample_rate = librosa.load(audio_path, sr=None)

# Calculate the maximum amplitude within the range [-0.8, 0.8]
current_max = np.max(np.abs(audio_data[(audio_data >= -0.8) & (audio_data <= 0.8)]))

# Desired maximum amplitude
desired_max = 0.9

# Calculate the amplification factor
amplification_factor = 30

# Amplify the audio by scaling the amplitude
amplified_audio = audio_data * amplification_factor

# Ensure the amplified audio does not exceed the range [-1, 1] (clip if necessary)
amplified_audio = np.clip(amplified_audio, -1, 1)  # Prevent clipping/distortion

# Calculate the number of samples to skip for the first 45 seconds
skip_samples = sample_rate * 45  # 45 seconds

# Skip the first 45 seconds of the audio
amplified_audio_cut = amplified_audio[skip_samples:]

# Display the waveform of the cut audio
plt.figure(figsize=(12, 6))
librosa.display.waveshow(amplified_audio_cut, sr=sample_rate, color='b')
plt.title('Amplified Audio Waveform (First 45 seconds cut)')
plt.show()

# Save the cut amplified audio file
output_path = 'amplified_audio_cut.wav'
sf.write(output_path, amplified_audio_cut, sample_rate)
