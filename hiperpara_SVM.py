from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
import parselmouth
from sklearn.model_selection import train_test_split, cross_val_score
import ast

def convert_feature_string(s):
    s = s.strip("[]")
    return [float(x) for x in s.split()]

df = pd.read_csv('audio_data_processed.csv')

df['Features'] = df['Features'].apply(convert_feature_string)

X = np.array(df['Features'].tolist())


X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'])

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# random search
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# hiperpara
param_distributions = {
    'C': [1, 10, 50, 68, 83, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf']
}

svm = SVC(random_state=2)

random_search = RandomizedSearchCV(
    svm,
    param_distributions=param_distributions,
    n_iter=12,  
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=2
)

random_search.fit(X_train_scaled, y_train)

print("Najlepsze parametry z Random Search:", random_search.best_params_)
print("Najlepsza dokładność walidacji krzyżowej:", random_search.best_score_)

# najlepszy model na zbiorze testowym
best_svm = random_search.best_estimator_
y_pred_random = best_svm.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_random)
print("Dokładność na zbiorze testowym:", test_accuracy)

# grid search
best_accuracy = 0
best_params = {}
for gamma in ['scale', 'auto']:
    for C in [1, 10, 50, 68, 83, 100]:
        classifier = SVC(C=C, gamma=gamma, kernel='rbf', random_state=2)
        classifier.fit(X_train_scaled, y_train)
        acc = classifier.score(X_test_scaled, y_test)
        print(f"Test - C={C}, gamma={gamma}, accuracy={acc}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_params = {'C': C, 'gamma': gamma}

print("Najlepsze parametry:", best_params)
print("Najlepsza dokładność testowa:", best_accuracy)

final_classifier = SVC(**best_params, kernel='rbf', random_state=2)
scores = cross_val_score(final_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Średnia dokładność kroswalidacji:", np.mean(scores))

final_classifier.fit(X_train_scaled, y_train)
y_pred = final_classifier.predict(X_test_scaled)

