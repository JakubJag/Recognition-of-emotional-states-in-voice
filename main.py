import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa

base_dir = 'Baza dźwiekowa'
sub_dirs = ['test', 'trening']

# Mel-Frequency Cepstral Coefficients
def extract_features(file_path, sample_rate=20000, n_mfcc=13):
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean

data = []

for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    for emotion in os.listdir(folder_path):
        emotion_folder = os.path.join(folder_path, emotion)
        if os.path.isdir(emotion_folder):  # sprawdzam czy to folder a nie plik
            for file in os.listdir(emotion_folder):
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_folder, file)
                    features = extract_features(file_path)
                    if features is not None:
                        data.append([features, emotion])  

df = pd.DataFrame(data, columns=['MFCC_Features', 'Emotion_Label'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = np.array([item for item in df['MFCC_Features']])
scaled_features = scaler.fit_transform(features)
df['MFCC_Features'] = list(scaled_features)

df.to_csv('audio_data.csv', index=False)
