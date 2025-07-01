import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from model import MSTRClassifier

SEQ_LEN       = 1
P             = 3
L             = 4
NUM_BLOCKS    = 3
DROPOUT       = 0.35
HIDDEN_DIMS   = (1024, 516)
NUM_HEADS     = 6
BATCH_SIZE    = 128
LR            = 1e-4
EPOCHS        = 100
CHECKPOINT    = "mstr_checkpoint.pth"

def convert_feature_string(s):
    return [float(x) for x in s.strip('[]').split()]

df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.vstack(df['Features'].values)
y = LabelEncoder().fit_transform(df['Emotion'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

n_features     = X_train.shape[1]
input_channels = n_features // SEQ_LEN
X_train_seq = X_train.reshape(-1, SEQ_LEN, input_channels)
X_test_seq  = X_test.reshape(-1, SEQ_LEN, input_channels)

class AudioSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dl = DataLoader(AudioSequenceDataset(X_train_seq, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(AudioSequenceDataset(X_test_seq,  y_test),  batch_size=BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MSTRClassifier(
    input_dim=input_channels,
    num_classes=len(np.unique(y)),
    p=P, L=L, num_blocks=NUM_BLOCKS,
    dropout=DROPOUT, hidden_dims=HIDDEN_DIMS, num_heads=NUM_HEADS
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for Xb, yb in train_dl:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, loss: {total_loss/len(train_dl.dataset):.4f}")

torch.save(model.state_dict(), CHECKPOINT)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_dl:
        Xb = Xb.to(device)
        all_preds.extend(model(Xb).argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

baseline_acc = accuracy_score(all_labels, all_preds)

importances = np.zeros(input_channels)
X_test_np = X_test_seq.copy()

def eval_acc(X_mod):
    ds = AudioSequenceDataset(X_mod, y_test)
    dl = DataLoader(ds, batch_size=BATCH_SIZE)
    preds, labs = [], []
    with torch.no_grad():
        for Xb, yb in dl:
            preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
            labs.extend(yb.numpy())
    return accuracy_score(labs, preds)

for f in range(input_channels):
    Xp = X_test_np.copy()
    flat = Xp[:, :, f].ravel()
    np.random.shuffle(flat)
    Xp[:, :, f] = flat.reshape(Xp.shape[0], SEQ_LEN)
    acc = eval_acc(Xp)
    importances[f] = baseline_acc - acc

ranking = np.argsort(importances)[::-1]

print(ranking.tolist())