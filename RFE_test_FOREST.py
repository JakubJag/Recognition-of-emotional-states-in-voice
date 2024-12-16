import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
import librosa

def extract_features(file_path, sample_rate=20000, n_mfcc=40): 
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)

    stats = [np.mean, np.median, np.std, np.min, np.max]
    features = np.hstack([
        np.hstack([stat(mfccs.T, axis=0) for stat in stats]),
        np.hstack([stat(chroma.T, axis=0) for stat in stats]),
        np.hstack([stat(spectral_contrast.T, axis=0) for stat in stats]),
        np.hstack([stat(rolloff.T, axis=0) for stat in stats]),
        np.hstack([stat(zero_crossings.T, axis=0) for stat in stats])
    ])
    return features

base_dir = 'Baza dźwiekowa'
sub_dirs = ['test', 'trening']

data = []
for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    for emotion in os.listdir(folder_path):
        emotion_folder = os.path.join(folder_path, emotion)
        if os.path.isdir(emotion_folder): 
            for file in os.listdir(emotion_folder):
                if file.endswith('.wav'):
                    file_path = os.path.join(emotion_folder, file)
                    features = extract_features(file_path)
                    if features is not None:
                        data.append([features, emotion])

df = pd.DataFrame(data, columns=['Features', 'Emotion'])
df['Emotion'] = df['Emotion'].str.lower()

X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'])

print("Rozkład etykiet w danych:", Counter(y))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=0)

rf = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=0)
rf.fit(X_train_scaled, y_train)

importances = rf.feature_importances_
important_features = np.argsort(importances)[::-1]

print("\nCechy posortowane według ważności:")
for idx in important_features:
    print(f"Cecha {idx}: Ważność = {importances[idx]:.4f}")

best_accuracy = 0
best_num_features = 0

print("\nTestowanie różnych liczby cech...\n")
for num_features in range(10, len(important_features) + 1, 10):
    selected_features = important_features[:num_features]
    model = RandomForestClassifier(
        n_estimators=500, 
        max_depth=30, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        random_state=0
    )
    model.fit(X_train_scaled[:, selected_features], y_train)
    y_pred = model.predict(X_test_scaled[:, selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Liczba cech: {num_features}, Dokładność: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = num_features

print(f"\nNajlepsza liczba cech: {best_num_features}, Dokładność: {best_accuracy:.4f}")