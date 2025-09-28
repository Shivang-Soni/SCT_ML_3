import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_images_from_folder(data_dir, target_size=(64, 64)):
    X = []
    y = []
    classes = {"Cats": 0, "Dogs": 1}
    for class_name, label in classes.items():
        folder_path = data_dir / class_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Ordner nicht gefunden: {folder_path}")
        for filename in os.listdir(folder_path):
            file_path = folder_path / filename
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    continue
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # optional Graustufen
                X.append(img.flatten())
                y.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return np.array(X), np.array(y)

def execute():
    # Basedirectory
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "raw"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load images
    X, y = load_images_from_folder(DATA_DIR, target_size=(64, 64))
    print(f"Loaded {X.shape[0]} images with {X.shape[1]} features each.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize SVM
    my_svm = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Debugging
    print("Unique classes in y_train:", set(y_train))
    print("Class distribution in y_train:", {label: list(y_train).count(label) for label in set(y_train)})

    # Train SVM
    print("Shivang's customised Support Vector Machine is being trained...")
    my_svm.fit(X_train, y_train)

    # Predictions
    y_pred = my_svm.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Achieved an accuracy of {accuracy:.4f}")

    print("\nHere is the classification report:")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png")
    plt.show()
