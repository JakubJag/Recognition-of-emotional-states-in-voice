import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import ast

def convert_feature_string(s):
    s = s.strip("[]")
    return [float(x) for x in s.split()]

df = pd.read_csv('audio_data_processed.csv')
df['Features'] = df['Features'].apply(convert_feature_string)
X = np.array(df['Features'].tolist())
y = np.array(df['Emotion'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RFE 
def rf_rfe(X, y, num_features_to_select):
    remaining_features = list(range(X.shape[1]))
    rankings = []

    while len(remaining_features) > num_features_to_select:
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=0,
            n_jobs=-1
        )
        model.fit(X[:, remaining_features], y)
        importances = model.feature_importances_
        least_important = np.argmin(importances)
        rankings.append(remaining_features.pop(least_important))

    rankings += remaining_features
    return rankings[::-1]

num_features_to_select = 50

# ranking
rankings = rf_rfe(X_train_scaled, y_train, num_features_to_select)
print("Ranking cech (od najważniejszych):", rankings)

# najlepsza liczba cech
best_accuracy = 0
best_num_features = 0

print("\nTestowanie różnych liczby cech \n")
for num_features in range(10, len(rankings) + 1, 10):
    selected_features = rankings[:num_features]
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0,
        n_jobs=-1
    )
    model.fit(X_train_scaled[:, selected_features], y_train)
    y_pred = model.predict(X_test_scaled[:, selected_features])
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Liczba cech: {num_features}, Dokładność: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = num_features

print(f"\nNajlepsza liczba cech: {best_num_features}, Dokładność: {best_accuracy:.4f}")

selected_features = rankings[:best_num_features]
final_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=-1
)
final_model.fit(X_train_scaled[:, selected_features], y_train)

y_pred = final_model.predict(X_test_scaled[:, selected_features])
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))