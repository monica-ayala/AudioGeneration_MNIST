import os
import pickle
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, ReLU, BatchNormalization, Flatten,
    Dense, Reshape, Conv2DTranspose, Activation, Lambda
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

tf.compat.v1.disable_eager_execution()

def build_encoder(input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
    encoder_input = Input(shape=input_shape)
    x = encoder_input
    for i, (filters, kernels, strides) in enumerate(zip(conv_filters, conv_kernels, conv_strides)):
        x = Conv2D(
            filters=filters,
            kernel_size=kernels,
            strides=strides,
            padding="same",
        )(x)
        x = ReLU(x)
        x = BatchNormalization(x)

    shape_before_bottleneck = K.int_shape(x)[1:]
    x = Flatten()(x)
    mu = Dense(latent_space_dim, name="mu")(x)
    log_variance = Dense(latent_space_dim)(x)

    def sample_point_from_normal_distribution(args):
        mu, log_variance = args
        epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_variance / 2) * epsilon

    encoder_output = Lambda(
        sample_point_from_normal_distribution
    )([mu, log_variance])

    return Model(encoder_input, encoder_output), shape_before_bottleneck, mu, log_variance


def build_decoder(
    latent_space_dim, shape_before_bottleneck, conv_filters, conv_kernels, conv_strides
):
    decoder_input = Input(shape=latent_space_dim)
    num_neurons = np.prod(shape_before_bottleneck)
    x = Dense(num_neurons)(decoder_input)
    x = Reshape(shape_before_bottleneck)(x)

    for i, (filters, kernels, strides) in reversed(list(enumerate(zip(conv_filters, conv_kernels, conv_strides)))):
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=kernels,
            strides=strides,
            padding="same"  
        )(x)
        x = ReLU(x)
        x = BatchNormalization(x)

    decoder_output = Conv2DTranspose(
        filters=1,
        kernel_size=conv_kernels[0]
        strides=(1, 1)
        padding="same"
    )(x)

    output_layer = Activation("sigmoid")(decoder_output)

    return Model(decoder_input, output_layer)


def build_autoencoder(encoder, decoder):
    model_input = encoder.input
    model_output = decoder(encoder(model_input))
    return Model(model_input, model_output)


def combined_loss(y_target, y_predicted, mu, log_variance, reconstruction_loss_weight):
    reconstruction_loss = K.mean(K.square(y_target - y_predicted), axis=[1, 2, 3])
    kl_loss = -0.5 * K.sum(1 + log_variance - K.square(mu) - K.exp(log_variance), axis=1)
    return reconstruction_loss_weight * reconstruction_loss + kl_loss

def save_autoencoder_weights(autoencoder, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    weights_path = os.path.join(folder, "weights.h5")
    autoencoder.save_weights(weights_path)

def load_autoencoder(autoencoder, folder):
    weights_path = os.path.join(folder, "weights.h5")
    autoencoder.load_weights(weights_path)

    return autoencoder

if __name__ == "__main__":
    input_shape = (256, 64, 1)
    conv_filters = (512, 256, 128, 64, 32)
    conv_kernels = (3, 3, 3, 3, 3)
    conv_strides = (2, 2, 2, 2, (2, 1))
    latent_space_dim = 128
    reconstruction_loss_weight = 1000000

    encoder, shape_before_bottleneck, mu, log_variance = build_encoder(
        input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim
    )

    decoder = build_decoder(
        latent_space_dim, shape_before_bottleneck, conv_filters, conv_kernels, conv_strides
    )

    autoencoder = build_autoencoder(encoder, decoder)

    optimizer = tf.keras.optimizers.legacy.Adam(0.0005)
    autoencoder.compile(
        optimizer=optimizer,
        loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, mu, log_variance, reconstruction_loss_weight),
        metrics=[MeanSquaredError()],
    )

    autoencoder.summary()
    encoder.summary()
    decoder.summary()

    directory_path = "/spectrograms"
    loaded_dict = {}

    file_list = os.listdir(directory_path)

    for file_name in file_list:
        if file_name.endswith('.npy'):
            file_path = os.path.join(directory_path, file_name)
            array = np.load(file_path)
            array_with_new_axis = np.expand_dims(array, axis=-1)
            loaded_dict[file_name] = array_with_new_axis
    
    keys = list(loaded_dict.keys())
    train_d = np.array([loaded_dict[key] for key in keys])
    
    " REMEMBER: YOU CAN TRAIN THE MODEL OR LOAD THE PRETRAINED WEIGHTS "
    # autoencoder = load_autoencoder(autoencoder, 'model'):

    history = autoencoder.fit(
        train_d,
        train_d,
        batch_size=64,
        epochs=150,
        shuffle=True
    )
    
    def inverse_normalize_spectrogram(normalized_spectrogram, min_val, max_val):
        return normalized_spectrogram * (max_val - min_val) + min_val

    def reconstruct_audio(stft, output_path, min_val, max_val):
        stft = stft[:, :, 0]
        magnitude_db = inverse_normalize_spectrogram(stft, min_val, max_val)
        magnitude_linear = librosa.db_to_amplitude(magnitude_db)
        threshold = np.percentile(np.abs(magnitude_linear), 97)  
        D_eliminated = np.where(np.abs(magnitude_linear) > threshold, 0, magnitude_linear)
        y_reconstructed = librosa.istft(D_eliminated, hop_length=256)
        sf.write(output_path, y_reconstructed, 22050)

        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_reconstructed, n_fft=512, hop_length=256)), ref=np.max)
        librosa.display.specshow(D, sr=22050, hop_length=256, x_axis='time', y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Reconstructed Spectrogram')
        plt.tight_layout()
        plt.show()

    def reconstruct(encoder, decoder, batch_of_images):
        latent_representations = encoder.predict(batch_of_images)
        reconstructed_images = decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    n_reconstructions = 5
    keys = list(loaded_dict.keys())  
    random_keys = np.random.choice(keys, n_reconstructions, replace=False)
    batch_of_images = np.array([loaded_dict[key] for key in random_keys])

    reconstructed_images, latent_representations = reconstruct(encoder, decoder, batch_of_images)

    def compare_reconstructed(reconstructed_images, min_max_dict, random_keys):
        for i, image in enumerate(reconstructed_images):
            file_name = random_keys[i]
            base_file_name = os.path.splitext(file_name)[0]

            min_val = min_max_dict[base_file_name]['min']
            max_val = min_max_dict[base_file_name]['max']

            reconstruct_audio(image, f"reconstructed_{file_name}.wav", min_val, max_val)
            reconstruct_audio(loaded_dict[file_name], f"original_{file_name}.wav", min_val, max_val)

    min_max_dict = np.load('min_max_values.npy', allow_pickle=True).item()

    compare_reconstructed(reconstructed_images, min_max_dict, random_keys)

    save_autoencoder_weights(autoencoder, 'model')
    