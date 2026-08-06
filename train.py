import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "C:/Users/anant/OneDrive/Desktop/final_year_project/SOCOFing" 
IMG_SIZE = (96, 96)
MODEL_SAVE_PATH = "fingerprint_verifier.weights.h5"

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_socofing_dataset(dataset_path):
    images, labels = [], []
    dataset_path = os.path.normpath(dataset_path)
    
    categories = {
        'Real': 0,
        'Altered/Altered-Easy': 1,
        'Altered/Altered-Medium': 1,
        'Altered/Altered-Hard': 1
    }

    if not os.path.exists(dataset_path):
        print(f"Error: Path '{dataset_path}' not found.")
        return None, None

    for rel_path, label in categories.items():
        full_path = os.path.join(dataset_path, rel_path.replace('/', os.sep))
        if not os.path.exists(full_path): continue
        
        print(f"Loading {rel_path}...")
        for img_name in os.listdir(full_path):
            if img_name.lower().endswith(('.bmp', '.png', '.jpg')):
                img = cv2.imread(os.path.join(full_path, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, IMG_SIZE))
                    labels.append(label)
    
    if not images: return None, None
    X = np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
    y = np.array(labels)
    return X, y

def start_training():
    X, y = load_socofing_dataset(DATASET_PATH)
    if X is None: return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    
    print("Training...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    model.save_weights(MODEL_SAVE_PATH)

 
    print("\n" + "="*30)
    print("  MODEL PERFORMANCE METRICS")
    print("="*30)
    
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Real', 'Altered']))

if __name__ == "__main__":
    start_training()