import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from itertools import product
from model import MSTRClassifier

SEQ_LEN     = 1
DEVICE      = torch.device(
    "mps"   if torch.backends.mps.is_available() else
    "cuda"  if torch.cuda.is_available()   else
    "cpu"
)
print("uzywane urzadzenie: ", DEVICE)

# ranking RFE z RFE_kolejnosc_MSTR
RANKING = [
    168,80,88,25,157,185,6,91,161,35,156,83,1,282,109,178,212,160,317,106,136,135,
    122,116,86,74,11,5,226,0,298,305,65,191,237,235,228,46,224,49,217,57,290,64,
    199,197,174,81,253,171,306,165,152,148,101,105,133,131,125,117,34,121,9,267,
    269,14,184,172,176,271,292,182,291,169,187,192,195,196,72,85,75,50,164,285,
    123,110,130,288,303,280,141,100,314,7,92,296,159,295,54,287,310,220,229,24,
    248,289,27,238,47,41,221,33,250,155,242,173,234,13,259,246,263
]
TOP_K = len(RANKING)

def convert_feature_string(s):
    return [float(x) for x in s.strip('[]').split()]

df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.vstack(df['Features'].values)
y = LabelEncoder().fit_transform(df['Emotion'])

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

idx = RANKING[:TOP_K]
X_train = X_train[:, idx].reshape(-1, SEQ_LEN, TOP_K)
X_val   = X_val[:,   idx].reshape(-1, SEQ_LEN, TOP_K)
X_test  = X_test[:,  idx].reshape(-1, SEQ_LEN, TOP_K)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_ds = SeqDataset(X_train, y_train)
val_ds   = SeqDataset(X_val,   y_val)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64)

# hiperpara
param_grid = {
    'lr':          [1e-3, 5e-4, 1e-4, 5e-5],
    'dropout':     [0.2, 0.35, 0.5],
    'hidden_dims': [(512,256), (1024,516), (2048,1024)],
    'num_heads':   [1, 2, 4, 6, 8],
    'num_blocks':  [2, 3, 4, 5],
    'p':           [2, 3, 4],
    'L':           [3, 4, 5],
}

# grid search
best_acc = 0.0
best_cfg = None

for (lr, dropout, hidden_dims, heads, blocks, p, L) in product(
        param_grid['lr'],
        param_grid['dropout'],
        param_grid['hidden_dims'],
        param_grid['num_heads'],
        param_grid['num_blocks'],
        param_grid['p'],
        param_grid['L']
    ):
    if TOP_K % heads != 0:
        continue

    model = MSTRClassifier(
        input_dim=TOP_K,
        num_classes=len(np.unique(y)),
        p=p, L=L, num_blocks=blocks,
        dropout=dropout, hidden_dims=hidden_dims,
        num_heads=heads
    ).to(DEVICE)
    opt  = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(DEVICE)
            p_out = model(Xb).argmax(dim=1).cpu().numpy()
            preds.extend(p_out)
            labs.extend(yb.numpy())
    acc = accuracy_score(labs, preds)

    print(f"lr={lr:.1e}, drop={dropout}, hid={hidden_dims}, heads={heads},"
          f" blocks={blocks}, p={p}, L={L} → val_acc={acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_cfg = dict(
            lr=lr, dropout=dropout, hidden_dims=hidden_dims,
            num_heads=heads, num_blocks=blocks, p=p, L=L
        )

print("\n najlepsze parametry: ")
print(best_cfg, " val_acc =", best_acc)


X_retrain = np.concatenate([X_train, X_val], axis=0)
y_retrain = np.concatenate([y_train, y_val], axis=0)
retrain_ds = SeqDataset(X_retrain, y_retrain)
retrain_loader = DataLoader(retrain_ds, batch_size=64, shuffle=True)
test_ds = SeqDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=64)

model = MSTRClassifier(
    input_dim=TOP_K,
    num_classes=len(np.unique(y)),
    **best_cfg
).to(DEVICE)
opt  = optim.AdamW(model.parameters(), lr=best_cfg['lr'], weight_decay=1e-4)
crit = nn.CrossEntropyLoss()

for epoch in range(30):
    model.train()
    for Xb, yb in retrain_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = model(Xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())
test_acc = accuracy_score(all_labels, all_preds)
print(f"\n Najlepsza dokładność testowa: {test_acc:.4f}")