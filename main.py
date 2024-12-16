import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter

base_dir = 'Baza dźwiekowa'
sub_dirs = ['test', 'trening']

import librosa
import numpy as np

def extract_features(file_path, sample_rate=20000, n_mfcc=40): 
    audio, sr = librosa.load(file_path, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)

    # Entropia spektralna
    spectral_entropy = -np.sum((librosa.amplitude_to_db(abs(librosa.stft(audio)))**2) * 
                               np.log(librosa.amplitude_to_db(abs(librosa.stft(audio)))**2 + 1e-10), axis=0)
    spectral_entropy = np.mean(spectral_entropy)

    # Entropia energetyczna
    energy = np.square(audio)
    frame_size = 512
    energy_frames = [np.sum(energy[i:i+frame_size]) for i in range(0, len(energy), frame_size)]
    energy_entropy = -np.sum((energy_frames / np.sum(energy_frames)) * 
                             np.log(energy_frames / np.sum(energy_frames) + 1e-10))

    # Statystyki cech
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


df.to_csv('audio_data_processed.csv', index=False)  # to przerzucić zeby było wczesniej bo nie ma znaczenia 

X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'])

print("Rozkład etykiet w danych:", Counter(y))
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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


# najwazniejsze cechy (to co zrobiłem w RFE_test_SVM)
important_features = [
    305, 303, 302, 289, 286, 282, 274, 272, 271, 265, 260, 248, 246, 245, 237, 235, 232, 228,
    227, 226, 224, 218, 207, 205, 172, 168, 162, 161, 160, 159, 157, 156, 141, 137, 134, 132,
    131, 130, 127, 121, 120, 114, 94, 82, 80, 45, 40, 36, 5, 0, 118, 288, 128, 87, 13, 187, 
    138, 231, 8, 113, 86, 53, 279, 196, 266, 70, 293, 77, 146, 56, 278, 234, 122, 163, 76, 
    123, 90, 151, 136, 11, 180, 89, 290, 182, 101, 142, 200, 42, 223, 179, 273, 241, 212, 
    211, 236, 247, 103, 100, 143, 198, 135, 285, 129, 270, 22, 230, 269, 140, 262, 83, 150, 
    281, 154, 148, 185, 295, 296, 102, 256, 21, 124, 221, 74, 10, 210, 52, 62, 263, 19, 95, 
    99, 171, 12, 84, 2, 125, 147, 35, 59, 34, 194, 204, 177, 219, 268, 220, 14, 202, 216, 
    173, 50, 144, 292, 225, 193, 75, 189, 110, 176, 306, 54, 43, 209, 304, 93, 178, 20, 240, 
    298, 297, 208, 239, 17, 280, 79, 153, 109, 254, 276, 275, 145, 85, 213, 78, 229, 57, 9, 
    188, 261, 217, 112, 39, 24, 214, 184, 91, 71, 242, 64, 60, 28, 44, 49, 4, 158, 92, 68, 
    152, 104, 174, 47, 167, 257, 133, 192, 267, 33, 203, 117, 301, 66, 1, 26, 23, 181, 46, 
    31, 294, 149, 116, 61, 166, 287, 107, 291, 249, 191, 88, 283, 244, 72, 169, 206, 65, 175, 
    222, 195, 170, 197, 32, 259, 139, 6, 55, 300, 183, 155, 238, 255, 299, 51, 105, 243, 69, 
    97, 164, 73, 25, 96, 41, 30, 253, 81, 233, 67, 108, 18, 251, 186, 277, 126, 38, 199, 284, 
    3, 29, 58, 115, 16, 111, 252, 264, 190, 165, 201, 215, 15, 63, 48, 37, 106, 98, 250, 258, 
    7, 27, 119
]
num_features_to_select = 250
selected_features = important_features[:num_features_to_select]

from sklearn.svm import SVC

X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

classifier = SVC(C=10, gamma='scale', kernel='rbf', random_state=2)
classifier.fit(X_train_selected, y_train)

y_pred = classifier.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Final Test Accuracy:", accuracy)

from sklearn.metrics import classification_report

# Zamiana numerów z powrotem na oryginalne etykiety
from sklearn.metrics import classification_report
target_names = label_encoder.inverse_transform(np.unique(y))
print(classification_report(y_test, y_pred, target_names=target_names))



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

# n_estimators_options = [50, 100, 200, 300]  
# max_depth_options = [None, 10, 20, 30]      
# min_samples_split_options = [2, 5, 10]      
# min_samples_leaf_options = [1, 2, 4]        

# best_accuracy = 0
# best_params = {}

# # Grid search
# for n_estimators in n_estimators_options:
#     for max_depth in max_depth_options:
#         for min_samples_split in min_samples_split_options:
#             for min_samples_leaf in min_samples_leaf_options:
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

# najwazniejsze cechy (to co zrobiłem w RFE_test_FOREST)
important_features = [
    160, 40, 0, 120, 161, 80, 6, 1, 5, 45, 279, 296, 41, 265, 172, 82, 10, 272, 46, 126, 
    50, 88, 4, 87, 293, 44, 180, 84, 12, 8, 13, 11, 135, 171, 51, 14, 52, 94, 96, 90, 54, 
    21, 173, 122, 174, 15, 124, 53, 48, 130, 170, 166, 133, 100, 179, 91, 93, 95, 99, 85, 
    102, 98, 9, 178, 295, 47, 56, 240, 61, 55, 260, 125, 123, 175, 16, 190, 57, 92, 301, 
    239, 20, 7, 128, 177, 101, 195, 127, 18, 3, 17, 221, 60, 196, 132, 109, 208, 131, 273, 
    83, 267, 168, 266, 106, 33, 243, 58, 245, 136, 183, 59, 73, 63, 194, 19, 169, 30, 193, 
    181, 264, 242, 129, 280, 138, 81, 49, 64, 187, 192, 263, 182, 137, 111, 191, 86, 89, 
    134, 75, 35, 43, 24, 210, 36, 209, 246, 189, 65, 176, 275, 2, 188, 162, 119, 112, 39, 
    198, 25, 103, 220, 67, 78, 29, 113, 107, 238, 97, 23, 185, 32, 68, 22, 241, 104, 147, 
    71, 110, 158, 76, 42, 159, 72, 274, 207, 227, 27, 200, 105, 38, 140, 219, 236, 199, 
    69, 269, 261, 74, 62, 139, 154, 197, 298, 151, 70, 237, 222, 217, 145, 211, 225, 244, 
    201, 114, 278, 184, 218, 31, 186, 299, 270, 215, 247, 116, 142, 118, 271, 277, 214, 
    231, 223, 294, 28, 37, 164, 165, 212, 26, 203, 288, 163, 206, 108, 144, 202, 205, 167, 
    66, 230, 148, 34, 291, 141, 153, 79, 276, 262, 155, 150, 157, 204, 146, 304, 143, 216, 
    233, 149, 297, 117, 228, 224, 303, 283, 289, 234, 268, 77, 229, 156, 226, 121, 115, 
    235, 287, 286, 213, 292, 302, 290, 281, 285, 152, 232, 282, 284, 300, 253, 254, 252, 
    251, 255, 250, 257, 256, 249, 248, 259, 258
]

num_features_to_select = 140 
selected_features = important_features[:num_features_to_select]

X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

final_rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=0)
final_rf_classifier.fit(X_train_selected, y_train)
y_pred_rf = final_rf_classifier.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred_rf)
accuracy = accuracy_score(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred_rf)
print("Dokładność:", accuracy)

from sklearn.metrics import classification_report
target_names = label_encoder.inverse_transform(np.unique(y))

print(classification_report(y_test, y_pred_rf, target_names=target_names))


plt.figure(figsize=(8, 6))
plt.bar(Counter(y).keys(), Counter(y).values())
plt.title("Rozkład etykiet emocji w danych")
plt.xlabel("Emocje")
plt.ylabel("Liczba próbek")
plt.show()


# augmentacja danych w celu lepszego wyrównania klas 
# crossvalidation 
# CNN-LSTM

# tSNE/PCA 
# clustering 

# KNN, Bayes
