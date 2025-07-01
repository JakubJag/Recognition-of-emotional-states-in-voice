import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

# random search 
param_distributions_rf = {
    'n_estimators': [50, 100, 150, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}

rf_classifier = RandomForestClassifier(random_state=42)

random_search_rf = RandomizedSearchCV(
    rf_classifier,
    param_distributions=param_distributions_rf,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search_rf.fit(X_train_scaled, y_train)

print("Najlepsze parametry z Random Search (RF):", random_search_rf.best_params_)
print("Najlepsza dokładność walidacji krzyżowej (RF):", random_search_rf.best_score_)

best_rf_random = random_search_rf.best_estimator_
y_pred_rf_random = best_rf_random.predict(X_test_scaled)
print("Dokładność na zbiorze testowym (RF Random Search):", accuracy_score(y_test, y_pred_rf_random))


# grid search
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search_rf = GridSearchCV(
    rf_classifier,
    param_grid=param_grid_rf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_rf.fit(X_train_scaled, y_train)

print("Najlepsze parametry z Grid Search (RF):", grid_search_rf.best_params_)
print("Najlepsza dokładność walidacji krzyżowej (RF):", grid_search_rf.best_score_)

best_rf_grid = grid_search_rf.best_estimator_
y_pred_rf_grid = best_rf_grid.predict(X_test_scaled)
print("Dokładność na zbiorze testowym (RF Grid Search):", accuracy_score(y_test, y_pred_rf_grid))

final_rf_classifier = RandomForestClassifier(**grid_search_rf.best_params_, random_state=42)
scores = cross_val_score(final_rf_classifier, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Średnia dokładność kroswalidacji (finalny RF):", np.mean(scores))

final_rf_classifier.fit(X_train_scaled, y_train)
y_pred_final = final_rf_classifier.predict(X_test_scaled)
print("Dokładność na zbiorze testowym (finalny RF):", accuracy_score(y_test, y_pred_final))