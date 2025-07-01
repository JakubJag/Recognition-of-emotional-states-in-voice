import numpy as np
import pandas as pd
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import parselmouth
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.mps
import torch.optim as optim
import time

base_dir = 'Baza dźwiekowa'
sub_dirs = ['test', 'trening']

# --------------------- Ekstrakcja cech --------------------- #
def extract_features(file_path, sample_rate=20000, n_mfcc=40): 

    audio, sr = librosa.load(file_path, sr=sample_rate)   
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y=audio)
    
    stft = librosa.stft(audio)
    db_stft = librosa.amplitude_to_db(np.abs(stft))
    spectral_entropy = -np.sum((db_stft**2) * np.log(db_stft**2 + 1e-10), axis=0)
    spectral_entropy = np.mean(spectral_entropy)
    
    energy = audio ** 2
    frame_size = 512
    energy_frames = np.array([np.sum(energy[i:i+frame_size])
                              for i in range(0, len(energy), frame_size)])
    energy_sum = np.sum(energy_frames) + 1e-10
    energy_entropy = -np.sum((energy_frames / energy_sum) *
                             np.log(energy_frames / energy_sum + 1e-10))
    
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    threshold = 0.02
    silent_parts = np.sum(energy_frames < threshold)
    
    f0 = librosa.yin(y=audio, fmin=librosa.note_to_hz('C2'),
                     fmax=librosa.note_to_hz('C7'))
    f0_mean = np.mean(f0)
    f0_std = np.std(f0)
    
    sound = parselmouth.Sound(file_path)
    pitch = sound.to_pitch() 
    intensity = sound.to_intensity() 
    pitch_mean = parselmouth.praat.call(pitch, "Get mean", 0.0, 0.0, "Hertz")
    pitch_min = np.nanmin(pitch.selected_array['frequency'])
    pitch_max = np.nanmax(pitch.selected_array['frequency'])
    intensity_mean = np.nanmean(intensity.values)    

    formants = sound.to_formant_burg()
    times = formants.xs()
    f1 = np.nanmean([formants.get_value_at_time(1, t) for t in times])
    f2 = np.nanmean([formants.get_value_at_time(2, t) for t in times])
    f3 = np.nanmean([formants.get_value_at_time(3, t) for t in times])
    
    pitch = sound.to_pitch()
    stats = [np.mean, np.median, np.std, np.min, np.max]
    mfccs_stats = np.concatenate([func(mfccs, axis=1) for func in stats])
    chroma_stats = np.concatenate([func(chroma, axis=1) for func in stats])
    spectral_contrast_stats = np.concatenate([func(spectral_contrast, axis=1) for func in stats])
    rolloff_stats = np.concatenate([func(rolloff, axis=1) for func in stats])
    zero_crossings_stats = np.concatenate([func(zero_crossings, axis=1) for func in stats])
    
    additional_features = np.array([
        spectral_entropy,
        energy_entropy,
        tempo,
        silent_parts,
        f0_mean,
        f0_std,
        f1,
        f2,
        f3,
        pitch_mean,      
        pitch_min,       
        pitch_max,       
        intensity_mean
    ])
    
    features = np.hstack([
        mfccs_stats,
        chroma_stats,
        spectral_contrast_stats,
        rolloff_stats,
        zero_crossings_stats,
        additional_features
    ])
    
    return features


# --------------------- Wizualizacja --------------------- #
def plot_confusion_matrix(cm, class_names, title="Macierz konfuzji", cmap="Blues"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Rzeczywiste etykiety')
    plt.xlabel('Przewidywane etykiety')
    plt.show()


def plot_classification_report(y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=report_df.index, y=report_df[metric])
        plt.title(f"{metric.capitalize()}")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Klasa")
        plt.xticks(rotation=45)
        plt.ylim(0, 1) 
        plt.show()


# ------------------------------------------------------------------------------------------------------------
# Komentuje na czas testowania  bo za długo zajmuje tworzenie całego CSV - audio_data_processed.csv
# data = []
# for sub_dir in sub_dirs:
#     folder_path = os.path.join(base_dir, sub_dir)
#     for emotion in os.listdir(folder_path):
#         emotion_folder = os.path.join(folder_path, emotion)
#         if os.path.isdir(emotion_folder):
#             for file in os.listdir(emotion_folder):
#                 if file.endswith('.wav'):
#                     file_path = os.path.join(emotion_folder, file)
#                     features = extract_features(file_path)
#                     if features is not None:
#                         data.append([features, emotion])

# df = pd.DataFrame(data, columns=['Features', 'Emotion'])
# df['Emotion'] = df['Emotion'].str.lower()
# df.to_csv('audio_data_processed.csv', index=False)
# ------------------------------------------------------------------------------------------------------------


# --------------------- Wczytanie danych --------------------- #
def convert_feature_string(s):
    s = s.strip("[]")
    return [float(x) for x in s.split()]


df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'])

X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------- SVM --------------------- #

# najwazniejsze cechy (to co zrobiłem w RFE_test_SVM)
important_features = [
    317, 316, 314, 313, 308, 307, 306, 305, 296, 295, 280, 274, 267, 260, 254, 246, 245,
    243, 240, 237, 217, 199, 191, 170, 160, 154, 149, 130, 127, 126, 123, 83, 82, 80, 78,
    75, 61, 57, 47, 45, 41, 40, 10, 9, 7, 5, 4, 2, 1, 0, 21, 42, 49, 91, 133, 58, 64, 120,
    288, 301, 144, 244, 220, 18, 8, 298, 87, 215, 70, 48, 134, 303, 153, 161, 234, 208, 131,
    24, 168, 164, 3, 81, 53, 97, 309, 227, 266, 165, 35, 163, 141, 184, 285, 186, 235, 231,
    156, 23, 290, 209, 221, 192, 310, 182, 150, 93, 22, 247, 62, 180, 11, 111, 183, 255, 51,
    107, 279, 270, 299, 31, 273, 167, 68, 140, 300, 171, 101, 284, 225, 138, 54, 146, 223,
    212, 211, 277, 187, 136, 36, 104, 169, 200, 195, 239, 174, 110, 196, 89, 74, 179, 178,
    143, 147, 213, 129, 66, 238, 302, 98, 224, 194, 198, 226, 132, 28, 92, 14, 16, 103, 151,
    19, 158, 207, 189, 33, 201, 73, 262, 271, 135, 312, 13, 56, 233, 79, 59, 293, 214, 90,
    257, 291, 34, 50, 84, 19, 181, 197, 102, 283, 116, 109, 236, 202, 118, 278, 125, 100,
    88, 311, 281, 222, 76, 145, 203, 228, 29, 15, 304, 55, 37, 26, 137, 117, 60, 230, 152,
    113, 250, 242, 63, 256, 176, 20, 206, 65, 115, 157, 106, 252, 190, 121, 276, 155, 177,
    297, 216, 287, 99, 286, 185, 172, 17, 292, 232, 72, 294, 162, 275, 229, 30, 282, 251,
    94, 95, 105, 38, 52, 188, 204, 128, 25, 269, 69, 12, 148, 46, 218, 6, 264, 173, 108,
    263, 142, 268, 43, 253, 265, 166, 27, 259, 261, 124, 122, 210, 71, 32, 39, 241, 112,
    193, 139, 219, 205, 85, 272, 44, 114, 289, 86, 96, 159, 175, 248, 249, 77, 67, 258, 315
]

num_features_to_select = 260
selected_features = important_features[:num_features_to_select]

from sklearn.svm import SVC

X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

start_time = time.time()

classifier = SVC(C=10, gamma='scale', kernel='rbf', random_state=2) # parametry dobrane z hiperpara_SVM.py
classifier.fit(X_train_selected, y_train)
svm_train_time = time.time() - start_time
print(f"Czas trenowania SVM: {svm_train_time:.2f} s")

y_pred = classifier.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Dokładność:", accuracy)


class_names = label_encoder.inverse_transform(np.unique(y))
plot_confusion_matrix(cm, class_names, title="Macierz konfuzji - SVM", cmap="Blues")

from sklearn.metrics import classification_report
target_names = label_encoder.inverse_transform(np.unique(y))
print(classification_report(y_test, y_pred, target_names=target_names))
plot_classification_report(y_test, y_pred, target_names)


# --------------------- Random Forest --------------------- #

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# najwazniejsze cechy (to co zrobiłem w RFE_test_FOREST)
important_features = [
    317, 316, 314, 305, 296, 293, 279, 272, 265, 260, 220, 196, 180, 174, 172, 161, 160, 135,
    126, 120, 109, 100, 96, 88, 87, 84, 82, 80, 51, 46, 45, 44, 41, 40, 21, 17, 16, 15, 14,
    13, 12, 11, 10, 9, 8, 6, 5, 4, 1, 0, 171, 24, 122, 33, 48, 50, 245, 7, 61, 94, 306, 20, 85,
    124, 54, 170, 52, 128, 240, 190, 273, 267, 313, 93, 55, 102, 179, 19, 195, 90, 53, 47,
    192, 221, 60, 178, 91, 263, 130, 125, 266, 83, 239, 208, 129, 30, 133, 18, 92, 166, 123,
    99, 175, 242, 199, 98, 95, 57, 193, 56, 35, 64, 25, 127, 81, 177, 222, 173, 168, 106,
    73, 59, 169, 101, 136, 241, 36, 194, 187, 23, 181, 138, 49, 63, 32, 243, 275, 264, 183, 301,
    72, 209, 62, 3, 116, 76, 185, 132, 119, 131, 311, 312, 280, 237, 58, 70, 158, 111, 197, 295,
    189, 89, 210, 201, 2, 198, 29, 162, 86, 105, 67, 107, 244, 246, 22, 71, 216, 75, 191, 206,
    134, 188, 308, 104, 271, 38, 31, 159, 140, 182, 212, 137, 278, 114, 238, 65, 118, 205, 43, 211,
    163, 110, 69, 151, 219, 97, 176, 274, 113, 213, 78, 207, 154, 112, 103, 165, 139, 261, 147,
    142, 39, 28, 42, 247, 66, 294, 184, 27, 34, 277, 144, 186, 224, 37, 217, 262, 145, 227, 229, 68, 150,
    200, 270, 167, 218, 231, 286, 74, 288, 203, 164, 298, 148, 236, 214, 115, 225, 269, 299, 146,
    276, 230, 152, 204, 121, 108, 223, 291, 77, 226, 309, 285, 292, 141, 26, 215, 153, 284, 281,
    79, 233, 268, 228, 302, 202, 287, 156, 149, 143, 289, 234, 157, 235, 304, 232, 117, 297, 155,
    283, 282, 310, 300, 290, 303, 307, 253, 252, 251, 254, 255, 250, 256, 248, 257, 249, 259, 258, 315
]

num_features_to_select = 190
selected_features = important_features[:num_features_to_select]

X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# parametry z hiperpara_FOREST.py
start_time = time.time()
final_rf_classifier = RandomForestClassifier(n_estimators=500, max_depth=30, min_samples_split=2, min_samples_leaf=1, random_state=0) 
final_rf_classifier.fit(X_train_selected, y_train)
rf_train_time = time.time() - start_time
print(f"Czas trenowania Random Forest: {rf_train_time:.2f} s")
y_pred_rf = final_rf_classifier.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred_rf)
accuracy = accuracy_score(y_test, y_pred_rf)
print("Confusion Matrix:\n", cm)

accuracy = accuracy_score(y_test, y_pred_rf)
print("Dokładność:", accuracy)

class_names = label_encoder.inverse_transform(np.unique(y))
plot_confusion_matrix(cm, class_names, title="Macierz konfuzji - Random Forest", cmap="Blues")

from sklearn.metrics import classification_report
target_names = label_encoder.inverse_transform(np.unique(y))

print(classification_report(y_test, y_pred_rf, target_names=target_names))
plot_classification_report(y_test, y_pred, target_names)


# --------------------- Improved NN --------------------- #
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = AudioDataset(X_train_selected, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = AudioDataset(X_test_selected, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Używane urządzenie:", device)

# definicja ulepszonej sieci     3 warstwy fully connected 
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, dropout_rate=0.5):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)  
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x

input_dim = X_train_selected.shape[1]   
hidden_dim1 = 256
hidden_dim2 = 128
num_classes = len(np.unique(y))
dropout_rate = 0.5

model_improved = ImprovedNN(input_dim, hidden_dim1, hidden_dim2, num_classes, dropout_rate).to(device)

criterion = nn.CrossEntropyLoss()   # karzemy model gdy sie myli nagradzamy gdy jest git 

optimizer = optim.AdamW(model_improved.parameters(), lr=0.0005, weight_decay=1e-5)  
# scheduler zmniejsza LR co 30 epok
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

num_epochs = 100
start_time = time.time()

for epoch in range(num_epochs):
    model_improved.train()
    running_loss = 0.0
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model_improved(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_features.size(0)
    epoch_loss = running_loss / len(train_dataset)
    scheduler.step()  # aktualizacja learning rate
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

nn_train_time = time.time() - start_time
print(f"Czas trenowania Improved NN: {nn_train_time:.2f} s")

model_improved.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        outputs = model_improved(batch_features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.numpy())

acc_improved = accuracy_score(all_labels, all_preds)
print("Dokładność (Improved NN):", acc_improved)
cm_improved = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm_improved, class_names, title="Macierz konfuzji - Improved NN", cmap="Blues")
print(classification_report(all_labels, all_preds, target_names=class_names))
plot_classification_report(all_labels, all_preds, target_names=class_names)



# --------------------- CNN-LSTM --------------------- #
seq_len = 2
feature_dim = X_train_selected.shape[1]

# sprawdzamy czy liczba cech sie dzieli przez seq_len jak nie to padding
if feature_dim % seq_len != 0:
    new_dim = feature_dim + (seq_len - feature_dim % seq_len)
    X_train_selected = np.hstack((X_train_selected, np.zeros((X_train_selected.shape[0], new_dim - feature_dim))))
    X_test_selected = np.hstack((X_test_selected, np.zeros((X_test_selected.shape[0], new_dim - feature_dim))))
    feature_dim = new_dim
    print("Nowa liczba cech po paddingu:", feature_dim)

input_channels = feature_dim // seq_len

X_train_seq = X_train_selected.reshape(-1, seq_len, input_channels)
X_test_seq = X_test_selected.reshape(-1, seq_len, input_channels)

class AudioSequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

batch_size = 32
train_dataset_seq = AudioSequenceDataset(X_train_seq, y_train)
test_dataset_seq = AudioSequenceDataset(X_test_seq, y_test)
train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True)
test_loader_seq = DataLoader(test_dataset_seq, batch_size=batch_size, shuffle=False)

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, kernel_size, lstm_hidden_size, lstm_layers, num_classes, dropout_rate):
        super(CNNLSTM, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=cnn_out_channels,
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.GRU(input_size=cnn_out_channels, hidden_size=lstm_hidden_size,
                           num_layers=lstm_layers, batch_first=True,
                           dropout=dropout_rate if lstm_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

cnn_lstm_model = CNNLSTM(
    input_channels=input_channels,
    cnn_out_channels=256,
    kernel_size=9,
    lstm_hidden_size=64,
    lstm_layers=1,
    num_classes=len(np.unique(y)),
    dropout_rate=0.4
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(cnn_lstm_model.parameters(), lr=0.0005)

num_epochs = 90
start_time = time.time()

for epoch in range(num_epochs):
    cnn_lstm_model.train()
    running_loss = 0.0
    for features, labels in train_loader_seq:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = cnn_lstm_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(train_dataset_seq)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

cnn_lstm_train_time = time.time() - start_time
print(f"Czas trenowania CNN-LSTM: {cnn_lstm_train_time:.2f} s")

cnn_lstm_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader_seq:
        features, labels = features.to(device), labels.to(device)
        outputs = cnn_lstm_model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc_cnn_lstm = accuracy_score(all_labels, all_preds)
print("Dokładność (CNN-LSTM):", acc_cnn_lstm)

cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, class_names, title="Macierz konfuzji - CNN-LSTM", cmap="Blues")
print(classification_report(all_labels, all_preds, target_names=class_names))
plot_classification_report(all_labels, all_preds, target_names=class_names)


# --------------------- MSTR --------------------- #
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from model import MSTRClassifier

RANKING = [
    168,80,88,25,157,185,6,91,161,35,156,83,1,282,109,178,212,160,317,106,136,135,
    122,116,86,74,11,5,226,0,298,305,65,191,237,235,228,46,224,49,217,57,290,64,
    199,197,174,81,253,171,306,165,152,148,101,105,133,131,125,117,34,121,9,267,
    269,14,184,172,176,271,292,182,291,169,187,192,195,196,72,85,75,50,164,285,
    123,110,130,288,303,280,141,100,314,7,92,296,159,295,54,287,310,220,229,24,
    248,289,27,238,47,41,221,33,250,155,242,173,234,13,259,246,263,96,95,293,94,
    37,274,166,26,149,247,8,104,150,307,239,38,153,154,40,42,236,43,198,219,256,
    213,252,60,210,209,301,284,115,113,203,44,268,21,190,230,308,181,107,108,79,
    270,225,193,315,103,2,4,111,102,20,99,48,19,18,61,62,63,118,66,69,73,93,309,
    32,76,22,77,87,89,90,29,143,200,243,201,183,273,206,275,175,208,211,120,216,
    147,146,264,286,223,245,255,138,244,258,128,299,132,254,30,207,241,28,251,257,
    68,67,214,58,240,15,55,16,52,36,23,17,265,71,59,312,170,300,112,129,142,144,
    145,189,304,277,311,84,177,186,180,10,202,12,126,114,31,188,249,134,137,139,
    167,222,98,297,163,276,316,179,53,51,218,278,3,266,313,272,281,261,279,283,
    262,158,151,233,232,231,45,227,215,124,205,204,127,302,82,97,260,140,119,294,
    162,39,70,56,78,194
]

SEQ_LEN    = 1
BATCH_SIZE = 128
EPOCHS     = 150

# hiperpara z grid search
HP = {
    'lr':          1e-3,
    'dropout':     0.35,
    'hidden_dims': (2048, 1024),
    'num_heads':   1,
    'num_blocks':  5,
    'p':           5,
    'L':           5,
}

DEVICE = torch.device(
    'mps' if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available()
    else 'cpu'
)
print("Using device:", DEVICE)

def convert_feature_string(s):
    return [float(x) for x in s.strip('[]').split()]

df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.vstack(df['Features'].values)
y = LabelEncoder().fit_transform(df['Emotion'])

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=420
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

#  top 263 cech
idx = RANKING
X_train = X_train[:, idx]
X_val   = X_val[:,   idx]
X_test  = X_test[:,  idx]

C = len(idx)
X_train_seq = X_train.reshape(-1, SEQ_LEN, C)
X_val_seq   = X_val  .reshape(-1, SEQ_LEN, C)
X_test_seq  = X_test .reshape(-1, SEQ_LEN, C)

class_weights_np = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)

class AudioSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(AudioSequenceDataset(X_train_seq, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(AudioSequenceDataset(X_val_seq,   y_val),
                          batch_size=BATCH_SIZE)
test_loader  = DataLoader(AudioSequenceDataset(X_test_seq,  y_test),
                          batch_size=BATCH_SIZE)

model = MSTRClassifier(
    input_dim=C,
    num_classes=len(np.unique(y)),
    p=HP['p'], L=HP['L'], num_blocks=HP['num_blocks'],
    dropout=HP['dropout'], hidden_dims=HP['hidden_dims'],
    num_heads=HP['num_heads']
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=HP['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    scheduler.step()
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS}, Loss: {avg_loss:.4f}")
    if epoch % 10 == 0:
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                p = model(Xb).argmax(1).cpu().numpy()
                preds.extend(p)
                labs.extend(yb.numpy())
        print(f"  >>> Val Acc: {accuracy_score(labs, preds):.4f}")
        model.train()

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(DEVICE)
        p = model(Xb).argmax(1).cpu().numpy()
        all_preds.extend(p)
        all_labels.extend(yb.numpy())

print("\nTest Acc:", accuracy_score(all_labels, all_preds))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class_names = LabelEncoder().fit(df['Emotion']).classes_

cm = confusion_matrix(all_labels, all_preds)
        
plot_confusion_matrix(cm, class_names, title="Macierz konfuzji - MSTR")
plot_classification_report(all_labels, all_preds, target_names=class_names)