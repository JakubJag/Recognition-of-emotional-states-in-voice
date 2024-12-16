import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter

def extract_features(file_path, sample_rate=20000, n_mfcc=40): 
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)

    spectral_entropy = -np.sum((librosa.amplitude_to_db(abs(librosa.stft(audio)))**2) * 
                               np.log(librosa.amplitude_to_db(abs(librosa.stft(audio)))**2 + 1e-10), axis=0)
    spectral_entropy = np.mean(spectral_entropy)

    energy = np.square(audio)
    frame_size = 512
    energy_frames = [np.sum(energy[i:i+frame_size]) for i in range(0, len(energy), frame_size)]
    energy_entropy = -np.sum((energy_frames / np.sum(energy_frames)) * 
                             np.log(energy_frames / np.sum(energy_frames) + 1e-10))

    stats = [np.mean, np.median, np.std, np.min, np.max]
    features = np.hstack([
        np.hstack([stat(mfccs.T, axis=0) for stat in stats]),
        np.hstack([stat(chroma.T, axis=0) for stat in stats]),
        np.hstack([stat(spectral_contrast.T, axis=0) for stat in stats]),
        np.hstack([stat(rolloff.T, axis=0) for stat in stats]),
        np.hstack([stat(zero_crossings.T, axis=0) for stat in stats]),
        spectral_entropy,
        energy_entropy
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

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Funkcje do RBF-RFE
def calculate_rbf_weights(X, y, model, gamma):
    """Oblicza wagi cech dla jądra RBF."""
    support_vectors = model.support_vectors_
    dual_coef = model.dual_coef_[0]

    weights = np.zeros(X.shape[1])
    for i, sv in enumerate(support_vectors):
        diff = X - sv  #
        rbf_grad = -2 * gamma * diff * np.exp(-gamma * np.sum(diff**2, axis=1, keepdims=True))
        weights += dual_coef[i] * rbf_grad.sum(axis=0)
    
    return np.abs(weights)

def rbf_rfe(X, y, gamma, num_features_to_select):
    """RFE dla SVM z jądrem RBF."""
    remaining_features = list(range(X.shape[1]))
    rankings = []

    while len(remaining_features) > num_features_to_select:
        model = SVC(kernel='rbf', C=10, gamma=gamma, random_state=2)
        model.fit(X[:, remaining_features], y)

        weights = calculate_rbf_weights(X[:, remaining_features], y, model, gamma)
        least_important = np.argmin(weights)
        rankings.append(remaining_features.pop(least_important))

    rankings += remaining_features
    return rankings[::-1]

gamma = 0.1  
num_features_to_select = 50  

rankings = rbf_rfe(X_train_scaled, y_train, gamma, num_features_to_select)
print("Ranking cech (od najważniejszych):", rankings)

best_accuracy = 0
best_num_features = 0

print("\nTestowanie różnych liczby cech...\n")
for num_features in range(10, len(rankings) + 1, 10):  
    selected_features = rankings[:num_features]
    model = SVC(C=10, gamma='scale', kernel='rbf', random_state=2)
    model.fit(X_train_scaled[:, selected_features], y_train)
    y_pred = model.predict(X_test_scaled[:, selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Liczba cech: {num_features}, Dokładność: {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = num_features

print(f"\nNajlepsza liczba cech: {best_num_features}, Dokładność: {best_accuracy:.4f}")

selected_features = rankings[:best_num_features]
final_model = SVC(C=10, gamma='scale', kernel='rbf', random_state=2)
final_model.fit(X_train_scaled[:, selected_features], y_train)

y_pred = final_model.predict(X_test_scaled[:, selected_features])
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))