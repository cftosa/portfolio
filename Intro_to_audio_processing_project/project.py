# COMP.SGN.120 Introduction to audio and speech processing
# Project
# Tommi Salonen and Joonas Kelavuori
# 09.12.2024
# The source folder path has to be modified to test the code

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import librosa as lb
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# All data was recorded in wav and we only used .wav files from freesound.org
def load_data(folder_path):
    samples = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            if file.startswith("car"):
                label = 0
            if file.startswith("tram"):
                label = 1

            input = os.path.join(folder_path, file)
            audio, sr = lb.load(input)
            samples.append((audio, sr, label))
    
    return samples

# Function to rename the samples downloaded from freesound used only once for tram samples and once for car samples
def rename(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            if file.startswith("77") or file.startswith("66"):
                name = "car_" + file
                input = os.path.join(folder_path, file)
                output = os.path.join(folder_path, name)
                os.rename(input, output)


# Split the samples for training, validation and testing. Using random to disjoint users.
def split_data(samples):
    random.seed(1)
    random.shuffle(samples)
    n = len(samples)

    train = samples[:int(0.8 * n)]  # 80%
    val = samples[int(0.8 * n):int(0.9 * n)]  # 10%
    test = samples[int(0.9 * n):]  # 10%

    return train, val, test

# Normalize the data
def normalize(data):
    max = np.max(data)
    normalized = data  / max 
    return normalized

# Extract 4 features
def extract_features(audio, sr):
    # Feture 1: RMS energy of the signal
    rms_energy = np.mean(lb.feature.rms(y=audio))

    # Feature 2: Spectral spread
    spectral_bandwidth = lb.feature.spectral_bandwidth(y=audio, sr=sr)
    mean_sb = np.mean(spectral_bandwidth)

    # Feature 3: Spectral centroid
    spectral_centroid = np.mean(lb.feature.spectral_centroid(y=audio, sr=sr))

    # Feature 4: DFT magnitudes in low, mid and high bands
    nfft = len(audio)
    stft = np.abs(lb.stft(audio, n_fft=nfft))
    
    freqs = lb.fft_frequencies(sr=sr, n_fft=nfft)
    
    low_band = (freqs >= 0) & (freqs < 1000)  # 0–1000 Hz
    mid_band = (freqs >= 1000) & (freqs < 3000)  # 1000–3000 Hz
    high_band = (freqs >= 3000) & (freqs < sr / 2)  # 3000 Hz–Nyquist
    
    low_energy = np.sum(stft[low_band, :])
    mid_energy = np.sum(stft[mid_band, :])
    high_energy = np.sum(stft[high_band, :])

    return (rms_energy, spectral_centroid, mean_sb, low_energy, mid_energy, high_energy)


# Use the normalize function to normalize all samples and calculate dft.
def normalize_and_features(samples):
    feature_data = []
    for sample in samples:
        data = sample[0]
        fs = sample[1]
        label = sample[2]

        normalized_data = normalize(data)
        features = extract_features(normalized_data, fs)
        
        feature_data.append((features, label))
    
    return feature_data

# Separate the data and the label to make using knn easier
def separate_labels(fft_data):
    data = []
    label = []
    for sample in fft_data:
        data.append(sample[0])
        label.append(sample[1])
    return data, label

# return only the data to be used in the model
def choose_feature(feature_data):
    chosen = 3                      # we chose to use the energy bands
    features, labels = zip(*feature_data)
    features = np.array(features)
    labels = np.array(labels)

    new_data = []
    for i in range(0, len(features)-1):
        new_data.append((features[i][chosen:], labels[i]))
    
    return new_data

# Plot the DFT of all samples in the same figure
def plot_all_samples_dft(samples):
    fft_data = []
    for sample in samples:
        audio = sample[0]
        sr = sample[1]
        label = sample[2]

        w = 4 * sr # 4 seconds
        frame = audio[len(audio)-w:] # use the last 4 seconds of the audio clip

        nfft = len(frame)
        f = fft(frame, n=nfft)
        m = np.abs(f)[:nfft // 2] * 2 / nfft

        fft_data.append((m, label))

    colors = ['red' if label == 0 else 'blue' for _, label in fft_data]

    plt.figure(figsize=(12, 6))
    for i, (fft_sample, label) in enumerate(fft_data):
        fs = samples[i][1]
        freqs = np.linspace(0, fs / 2, len(fft_sample), endpoint=False)
        
        plt.plot(freqs, fft_sample, color=colors[i], alpha=0.5, linewidth=0.8)

    plt.title("DFTs of all samples")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot([], [], color='red', label='Car (Label 0)')
    plt.plot([], [], color='blue', label='Tram (Label 1)')
    plt.legend(loc='upper right')

    plt.show()

# Plot the pointcloud used by knn
def plot_knn_data(x_train, y_train):
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    if x_train.shape[1] > 2:
        pca = PCA(n_components=2)
        x_train = pca.fit_transform(x_train)

    color_map = {0: 'red', 1: 'blue'}
    colors = [color_map[label] for label in y_train]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_train[:, 0], x_train[:, 1], c=colors, alpha=0.8, edgecolor='k')

    plt.title("kNN: Pointcloud from training data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(alpha=0.3)

    plt.scatter([], [], color='red', label='Car (Label 0)')
    plt.scatter([], [], color='blue', label='Tram (Label 1)')
    plt.legend(loc="best")

    plt.show()

# Plot all the features in a single figure to determine which of them to use
def plot_features(feature_data):
    features, labels = zip(*feature_data)
    features = np.array(features)
    labels = np.array(labels)
    
    feature_names = [
        "RMS Energy", 
        "Spectral Centroid", 
        "Spectral Bandwidth", 
        "Energy Bands (Low, Mid, High)"
    ]
    band_names = ["Low Band Energy", "Mid Band Energy", "High Band Energy"]
    
    # Classes, their names and colors
    classes = [0, 1]
    class_names = ["Car", "Tram"]
    colors = ["blue", "red"]
    
    plt.figure(figsize=(15, 8))

    for i, feature_name in enumerate(feature_names[:3]):  # Features 1, 2 and 3
        plt.subplot(2, 2, i + 1)
        means = []
        stds = []
        
        for label in classes:
            class_features = features[labels == label, i]
            means.append(np.mean(class_features))
            stds.append(np.std(class_features))
        
        x = np.arange(len(classes))
        plt.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5, tick_label=class_names)
        
        plt.title(feature_name)
        plt.ylabel("Mean Value")
        plt.xlabel("Class")
        plt.xticks(ticks=x, labels=class_names)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot feature 4
    plt.subplot(2, 2, 4)
    width = 0.3
    for band_idx, band_name in enumerate(band_names):
        means = []
        stds = []
        
        for label in classes:
            class_features = features[labels == label, 3 + band_idx]
            means.append(np.mean(class_features))
            stds.append(np.std(class_features))
        
        x = np.arange(len(classes)) + band_idx * width
        plt.bar(x, means, yerr=stds, width=width, color=colors, alpha=0.7, capsize=5)
    
    plt.title("Energy Bands")
    plt.ylabel("Mean Value")
    plt.xlabel("Class")
    plt.xticks(ticks=np.arange(len(classes)) + width, labels=class_names)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()



def main():
    # Change folder path to the folder where samples are located
    folder_path = "C:/Users/salon/IntroToAudioProcessing/Project/samples"
    samples = load_data(folder_path)

    feature_data = normalize_and_features(samples)
    data = choose_feature(feature_data)
    train, val, test = split_data(data)

    x_train, y_train = separate_labels(train)
    x_val, y_val = separate_labels(val)
    x_test, y_test = separate_labels(test)
    
    knn = KNeighborsClassifier(n_neighbors=9) # n_neighbors was tested with all odd numbers from 1 to 15
    knn.fit(x_train, y_train)
    y_pred_v = knn.predict(x_val)
    
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred_v, cmap=plt.cm.Blues)
    plt.title("Confusion matrix from validation data")
    plt.show()
    
    y_pred_t = knn.predict(x_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_t, cmap=plt.cm.Blues)
    plt.title("Confusion matrix from test data")
    plt.show()
    
    accuracy_v = accuracy_score(y_val, y_pred_v)
    precision_v = precision_score(y_val, y_pred_v)
    recall_v = recall_score(y_val, y_pred_v)
    print(f"Accuracy for validation data: {accuracy_v:.5f}")
    print(f"Precision for validation data: {precision_v:.5f}")
    print(f"Recall for validation data: {recall_v:.5f}")
    
    accuracy_t = accuracy_score(y_test, y_pred_t)
    precision_t = precision_score(y_test, y_pred_t)
    recall_t = recall_score(y_test, y_pred_t)
    print(f"Accuracy for test data: {accuracy_t:.5f}")
    print(f"Precision for test data: {precision_t:.5f}")
    print(f"Recall for test data: {recall_t:.5f}")
    
    plot_all_samples_dft(samples)
    plot_knn_data(x_train, y_train)
    plot_features(feature_data)
    
    

if __name__ == "__main__":
    main()
