import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

base_dir = 'Baza dźwiekowa'
sub_dirs = ['test', 'trening']

def extract_features(file_path, sample_rate=20000, n_mfcc=40): 
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.mean(chroma.T, axis=0),
        np.mean(spectral_contrast.T, axis=0),
        np.mean(rolloff.T, axis=0),
        np.mean(zero_crossings.T, axis=0)
    ])
    return features

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# best_accuracy = 0
# best_params = {}
# for gamma in ['scale', 'auto']:
#     for C in [1, 10, 50, 83, 100]:
#         classifier = SVC(C=C, gamma=gamma, kernel='rbf', random_state=2)
#         classifier.fit(X_train_scaled, y_train)
#         acc = classifier.score(X_test_scaled, y_test)
#         print(f"Test - C={C}, gamma={gamma}, accuracy={acc}")
#         if acc > best_accuracy:
#             best_accuracy = acc
#             best_params = {'C': C, 'gamma': gamma}

# print("Najlepsze parametry:", best_params)
# print("Najlepsza dokładność testowa:", best_accuracy)

# final_classifier = SVC(**best_params, kernel='rbf', random_state=2)
# scores = cross_val_score(final_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
# print("Średnia dokładność kroswalidacji:", np.mean(scores))

# final_classifier.fit(X_train_scaled, y_train)
# y_pred = final_classifier.predict(X_test_scaled)

classifier = SVC(C=10, gamma='scale', kernel='rbf', random_state=2)
classifier.fit(X_train_scaled, y_train)
y_pred = classifier.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Final Test Accuracy:", accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=np.unique(y)))

df.to_csv('audio_data_processed.csv', index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

# n_estimators_options = [50, 100, 200, 300]  # liczba drzew
# max_depth_options = [None, 10, 20, 30]      # maksymalna głębokość drzewa
# min_samples_split_options = [2, 5, 10]      # minimalna liczba próbek do podziału
# min_samples_leaf_options = [1, 2, 4]        # minimalna liczba próbek na liść

# best_accuracy = 0
# best_params = {}

# # Grid search
# for n_estimators in n_estimators_options:
#     for max_depth in max_depth_options:
#         for min_samples_split in min_samples_split_options:
#             for min_samples_leaf in min_samples_leaf_options:
#                 # Tworzenie modelu z danym zestawem parametrów
#                 rf_classifier = RandomForestClassifier(
#                     n_estimators=n_estimators,
#                     max_depth=max_depth,
#                     min_samples_split=min_samples_split,
#                     min_samples_leaf=min_samples_leaf,
#                     random_state=42
#                 )

#                 scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
#                 mean_score = np.mean(scores)
                
#                 print(f"Test - n_estimators={n_estimators}, max_depth={max_depth}, "
#                       f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
#                       f"accuracy={mean_score}")
                
#                 if mean_score > best_accuracy:
#                     best_accuracy = mean_score
#                     best_params = {
#                         'n_estimators': n_estimators,
#                         'max_depth': max_depth,
#                         'min_samples_split': min_samples_split,
#                         'min_samples_leaf': min_samples_leaf
#                     }

# print("\nNajlepsze parametry:", best_params)
# print("Najlepsza dokładność walidacji:", best_accuracy)

#{'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1}

final_rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=0)
final_rf_classifier.fit(X_train_scaled, y_train)
y_pred_rf = final_rf_classifier.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_rf)
accuracy = accuracy_score(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred_rf)
print("Dokładność:", accuracy)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf, target_names=np.unique(y)))

plt.figure(figsize=(8, 6))
plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Rozkład etykiet emocji w danych")
plt.xlabel("Emocje")
plt.ylabel("Liczba próbek")
plt.show()


# sprawdzic czy nie lepsze beda innne cechy podane  
#  PCA 
# tSNE wykres