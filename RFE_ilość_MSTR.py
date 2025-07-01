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

SEQ_LEN     = 1
P           = 3
L           = 4
NUM_BLOCKS  = 3
DROPOUT     = 0.35
HIDDEN_DIMS = (1024, 516)
MAX_HEADS   = 6    
BATCH_SIZE  = 128
LR          = 1e-4
EPOCHS      = 50
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def convert_feature_string(s):
    return [float(x) for x in s.strip('[]').split()]

df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.vstack(df['Features'].values)
y = LabelEncoder().fit_transform(df['Emotion'])

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full  = scaler.transform(X_test_full)

ranking = [
    168, 80, 88, 25, 157, 185, 6, 91, 161, 35, 156, 83, 1, 282, 109, 178, 212,
    160, 317, 106, 136, 135, 122, 116, 86, 74, 11, 5, 226, 0, 298, 305, 65, 191,
    237, 235, 228, 46, 224, 49, 217, 57, 290, 64, 199, 197, 174, 81, 253, 171,
    306, 165, 152, 148, 101, 105, 133, 131, 125, 117, 34, 121, 9, 267, 269, 14,
    184, 172, 176, 271, 292, 182, 291, 169, 187, 192, 195, 196, 72, 85, 75, 50,
    164, 285, 123, 110, 130, 288, 303, 280, 141, 100, 314, 7, 92, 296, 159, 295,
    54, 287, 310, 220, 229, 24, 248, 289, 27, 238, 47, 41, 221, 33, 250, 155,
    242, 173, 234, 13, 259, 246, 263, 96, 95, 293, 94, 37, 274, 166, 26, 149,
    247, 8, 104, 150, 307, 239, 38, 153, 154, 40, 42, 236, 43, 198, 219, 256,
    213, 252, 60, 210, 209, 301, 284, 115, 113, 203, 44, 268, 21, 190, 230, 308,
    181, 107, 108, 79, 270, 225, 193, 315, 103, 2, 4, 111, 102, 20, 99, 48, 19,
    18, 61, 62, 63, 118, 66, 69, 73, 93, 309, 32, 76, 22, 77, 87, 89, 90, 29,
    143, 200, 243, 201, 183, 273, 206, 275, 175, 208, 211, 120, 216, 147, 146,
    264, 286, 223, 245, 255, 138, 244, 258, 128, 299, 132, 254, 30, 207, 241,
    28, 251, 257, 68, 67, 214, 58, 240, 15, 55, 16, 52, 36, 23, 17, 265, 71,
    59, 312, 170, 300, 112, 129, 142, 144, 145, 189, 304, 277, 311, 84, 177,
    186, 180, 10, 202, 12, 126, 114, 31, 188, 249, 134, 137, 139, 167, 222, 98,
    297, 163, 276, 316, 179, 53, 51, 218, 278, 3, 266, 313, 272, 281, 261, 279,
    283, 262, 158, 151, 233, 232, 231, 45, 227, 215, 124, 205, 204, 127, 302,
    82, 97, 260, 140, 119, 294, 162, 39, 70, 56, 78, 194
]

def pick_num_heads(channels, max_heads):
    for h in range(max_heads, 0, -1):
        if channels % h == 0:
            return h
    return 1

best_acc = 0.0
best_n   = 0
best_heads = None

for n_feats in range(10, len(ranking)+1, 10):
    top_idx = ranking[:n_feats]
    X_tr = X_train_full[:, top_idx]
    X_te = X_test_full[:,  top_idx]

    ch = X_tr.shape[1] // SEQ_LEN
    heads = pick_num_heads(ch, MAX_HEADS)

    X_tr_seq = X_tr.reshape(-1, SEQ_LEN, ch)
    X_te_seq = X_te.reshape(-1, SEQ_LEN, ch)

    class SeqDS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
        def __len__(self):  return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    train_dl = DataLoader(SeqDS(X_tr_seq, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(SeqDS(X_te_seq,  y_test),  batch_size=BATCH_SIZE)

    model = MSTRClassifier(
        input_dim=ch, num_classes=len(np.unique(y)),
        p=P, L=L, num_blocks=NUM_BLOCKS,
        dropout=DROPOUT, hidden_dims=HIDDEN_DIMS,
        num_heads=heads
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    for _ in range(EPOCHS):
        model.train()
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            p = model(Xb.to(DEVICE)).argmax(1).cpu().numpy()
            preds.extend(p); labs.extend(yb.numpy())

    acc = accuracy_score(labs, preds)
    print(f"{n_feats:3d} cech — heads={heads:2d} → acc = {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_n   = n_feats
        best_heads = heads

print("\n najlepsza konifguracja:", best_n, "cech / glow - ", best_heads,
      "dokladnosc testowa =", best_acc)
print("ranking cech:", ranking[:best_n])