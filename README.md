projekt koncentruje się na rozpoznawaniu emocji w nagraniach głosowych za pomocą metod uczenia maszynowego.
W ramach projektu zastosowano takie modele jak SVM i Random Forest oraz przeprowadzono selekcję cech przy użyciu RFE (Recursive Feature Elimination).

1. main.py: Główny skrypt wykonujący pełen proces:
	•	Przetwarzanie danych audio i ekstrakcję cech.
	•	Trenowanie modeli klasyfikacyjnych (SVM i Random Forest).
	•	Wyświetlanie wyników w postaci macierzy pomyłek i raportów klasyfikacyjnych.

2. RFE_test_FOREST.py:
	•	Oblicza najważniejsze cechy dla modelu Random Forest.
	•	Testuje różne liczby cech, aby znaleźć ich optymalną ilość pod kątem dokładności klasyfikacji.

4. RFE_test_SVM.py:
	•	Oblicza najważniejsze cechy dla modelu SVM z jądrem RBF.
	•	Testuje różne liczby cech, aby zoptymalizować wydajność modelu.

5. audio_data.csv: Surowe dane audio w formacie CSV.

6. audio_data_processed.csv: Dane przetworzone z wyekstrahowanymi cechami, gotowe do użycia w modelach.
  
7. Baza dźwiekowa/: Folder zawierający dane audio (rozszerzenie .wav) podzielone na podfoldery test i trening.
