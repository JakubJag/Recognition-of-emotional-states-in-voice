import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def convert_feature_string(s):
    s = s.strip("[]")
    return [float(x) for x in s.split()]

df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'].tolist())

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

seq_len = 2
feature_dim = X_train_scaled.shape[1]

if feature_dim % seq_len != 0:
    new_dim = feature_dim + (seq_len - (feature_dim % seq_len))
    padding = np.zeros((X_train_scaled.shape[0], new_dim - feature_dim))
    X_train_scaled = np.hstack((X_train_scaled, padding))

    padding = np.zeros((X_test_scaled.shape[0], new_dim - feature_dim))
    X_test_scaled = np.hstack((X_test_scaled, padding))

    feature_dim = new_dim  
    print("Nowa liczba cech po dodaniu zer:", feature_dim)

input_channels = feature_dim // seq_len  
X_train_seq = X_train_scaled.reshape(-1, seq_len, input_channels)
X_test_seq = X_test_scaled.reshape(-1, seq_len, input_channels)

X_train_seq, X_val_seq, y_train, y_val = train_test_split(
    X_train_seq, y_train, test_size=0.2, random_state=42, stratify=y_train
)

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
val_dataset_seq = AudioSequenceDataset(X_val_seq, y_val)
test_dataset_seq = AudioSequenceDataset(X_test_seq, y_test)

train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True)
val_loader_seq = DataLoader(val_dataset_seq, batch_size=batch_size, shuffle=False)
test_loader_seq = DataLoader(test_dataset_seq, batch_size=batch_size, shuffle=False)

class CNNLSTM(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, kernel_size, lstm_hidden_size, lstm_layers, num_classes,
                 dropout_rate):
        super(CNNLSTM, self).__init__()
        # CNN 
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=cnn_out_channels,
                              kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  

        # LSTM 
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

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Używane urządzenie:", device)

# Random Search
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return best_val_acc

# hiperpara
param_grid = {
    'cnn_out_channels': [16, 32, 64, 128, 256, 512],
    'kernel_size': [3, 5, 7, 9, 11, 13, 15],
    'lstm_hidden_size': [32, 64, 128, 256, 512],
    'lstm_layers': [1, 2, 3, 4],
    'dropout_rate': [0.2, 0.3, 0.4, 0.5, 0.7],
    'learning_rate': [1e-3, 5e-4, 1e-4, 5e-5, 1e-6]
}
num_classes = len(np.unique(y_train))
num_search_iterations = 100 
num_epochs_search = 20  

best_params = None
best_val_acc = 0.0

for i in range(num_search_iterations):

    params = {key: random.choice(values) for key, values in param_grid.items()}
    print(f"\nIteracja {i + 1}, wybrane parametry: {params}")

    model = CNNLSTM(
        input_channels=input_channels,
        cnn_out_channels=params['cnn_out_channels'],
        kernel_size=params['kernel_size'],
        lstm_hidden_size=params['lstm_hidden_size'],
        lstm_layers=params['lstm_layers'],
        num_classes=num_classes,
        dropout_rate=params['dropout_rate']
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])

    val_acc = train_model(model, train_loader_seq, val_loader_seq, criterion, optimizer, num_epochs_search, device)
    print(f"Iteracja {i + 1}, Val Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = params

print("\nNajlepsze hiperparametry:", best_params)
print("Najlepsza dokładność walidacji:", best_val_acc)

# trenowanie finalnego modelu 
X_combined_seq = np.concatenate([X_train_seq, X_val_seq], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)
combined_dataset = AudioSequenceDataset(X_combined_seq, y_combined)
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

final_model = CNNLSTM(
    input_channels=input_channels,
    cnn_out_channels=best_params['cnn_out_channels'],
    kernel_size=best_params['kernel_size'],
    lstm_hidden_size=best_params['lstm_hidden_size'],
    lstm_layers=best_params['lstm_layers'],
    num_classes=num_classes,
    dropout_rate=best_params['dropout_rate']
)
final_model = final_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(final_model.parameters(), lr=best_params['learning_rate'])
num_epochs_final = 90  # Większa liczba epok dla finalnego treningu DLA 90 SPROBOWAC CHYBA NAJMNIEJSZY BŁĄD

print("\nTrening finalnego modelu...")
for epoch in range(num_epochs_final):
    final_model.train()
    running_loss = 0.0
    for features, labels in combined_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = final_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
    epoch_loss = running_loss / len(combined_loader.dataset)
    print(f"Final Model Epoch {epoch + 1}/{num_epochs_final}, Loss: {epoch_loss:.4f}")

final_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for features, labels in test_loader_seq:
        features, labels = features.to(device), labels.to(device)
        outputs = final_model(features)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
test_acc = accuracy_score(all_labels, all_preds)
print("\nDokładność na zbiorze testowym (Final Model):", test_acc)

# Dokładność na zbiorze testowym (Final Model): 0.8537735849056604 dla num_epochs_final - 90, z hiperparametrami {'cnn_out_channels': 256, 'kernel_size': 9, 'lstm_hidden_size': 64, 'lstm_layers': 1, 'dropout_rate': 0.4, 'learning_rate': 0.0005}


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



cm = confusion_matrix(all_labels, all_preds)
class_names = label_encoder.classes_  

plot_confusion_matrix(cm, class_names, title="Macierz konfuzji - Final CNN-LSTM", cmap="Blues")

plot_classification_report(all_labels, all_preds, target_names=class_names)