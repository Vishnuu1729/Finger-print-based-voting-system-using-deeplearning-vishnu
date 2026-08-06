import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import filedialog

# Constants
IMG_SIZE = (96, 96)
MODEL_PATH = "fingerprint_verifier.weights.h5"

def robust_imread(path, flags=cv2.IMREAD_GRAYSCALE):
    """
    Reads an image from a path that may contain Unicode/Japanese characters.
    """
    try:
        if not os.path.exists(path):
            return None
        # Read file as a byte array to bypass Windows encoding issues
        with open(path, 'rb') as f:
            chunk = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(chunk, flags)
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def get_model():
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
    
    if os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        return model
    else:
        # Fallback for demonstration if weights aren't there yet
        print(f"Warning: Weights file '{MODEL_PATH}' not found. Using uninitialized model for UI testing.")
        return model

def display_visual_result(name, img, result, confidence, reason):
    canvas = np.zeros((400, 850, 3), dtype=np.uint8) + 30 
    
    display_img = cv2.resize(img, (300, 300))
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    canvas[50:350, 50:350] = display_img
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_main = (255, 255, 255)
    color_accent = (241, 102, 99) if result == "ALTERED" else (94, 197, 34) 
    
    cv2.putText(canvas, f"TEST: {name}", (380, 80), font, 0.8, color_main, 2)
    cv2.putText(canvas, f"RESULT: {result}", (380, 130), font, 1, color_accent, 3)
    cv2.putText(canvas, f"CONFIDENCE: {confidence:.2%}", (380, 180), font, 0.7, color_main, 1)
    
    y_offset = 240
    cv2.putText(canvas, "REASON:", (380, y_offset), font, 0.6, (150, 150, 150), 2)
    y_offset += 30
    
    words = reason.split(' ')
    line = ""
    for word in words:
        if len(line + word) < 45:
            line += word + " "
        else:
            cv2.putText(canvas, line, (380, y_offset), font, 0.5, (200, 200, 200), 1)
            line = word + " "
            y_offset += 25
    cv2.putText(canvas, line, (380, y_offset), font, 0.5, (200, 200, 200), 1)

    cv2.putText(canvas, "Press any key to see next test...", (50, 385), font, 0.4, (100, 100, 100), 1)
    
    cv2.imshow("Nexus AI - Robustness Analysis", canvas)
    cv2.waitKey(0)

def display_final_verdict(scores_dict, filename):
    canvas = np.zeros((520, 850, 3), dtype=np.uint8) + 20 
    
    weights = {
        "Baseline (Original)": 3.0,
        "Rotation (90 deg)": 1.0,
        "Brightness (+30)": 1.0,
        "Blur (Gaussian 5x5)": 1.0,
        "Inverted Colors": 1.0
    }
    
    total_weight = sum(weights.values())
    weighted_sum = 0
    
    for test, weight in weights.items():
        weighted_sum += (scores_dict.get(test, 0.5) * weight)
    
    final_score = weighted_sum / total_weight
    final_verdict = "REAL" if final_score < 0.5 else "ALTERED"
    security_confidence = (1 - final_score) if final_score < 0.5 else final_score
    verdict_color = (94, 197, 34) if final_verdict == "REAL" else (241, 102, 99)
    color_main = (255, 255, 255)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "FINAL SYSTEM VERDICT", (250, 60), font, 1.2, color_main, 3)
    cv2.putText(canvas, f"File: {filename}", (50, 110), font, 0.6, (150, 150, 150), 1)
    cv2.line(canvas, (50, 130), (800, 130), (50, 50, 50), 2)
    
    cv2.putText(canvas, "Authenticity Score:", (50, 180), font, 0.7, (200, 200, 200), 2)
    bar_width = 600
    cv2.rectangle(canvas, (50, 210), (50 + bar_width, 240), (40, 40, 40), -1)
    filled_width = int(bar_width * (1 - final_score))
    cv2.rectangle(canvas, (50, 210), (50 + filled_width, 240), verdict_color, -1)
    cv2.putText(canvas, f"{security_confidence:.2%}", (50 + bar_width + 10, 230), font, 0.6, color_main, 1)

    cv2.rectangle(canvas, (50, 310), (800, 430), (30, 30, 30), -1)
    cv2.rectangle(canvas, (50, 310), (800, 430), verdict_color, 2)
    
    verdict_text = "RESULT: Authentic Fingerprint" if final_verdict == "REAL" else "RESULT: Altered / Fake Detected"
    cv2.putText(canvas, verdict_text, (80, 385), font, 1.0, verdict_color, 3)
    
    cv2.putText(canvas, "Press any key to close the security report...", (280, 490), font, 0.5, (100, 100, 100), 1)
    
    cv2.imshow("Nexus AI - Final Verdict", canvas)
    cv2.waitKey(0)

def run_inference(name, img, model):
    processed = cv2.resize(img, IMG_SIZE).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
    pred = model.predict(processed, verbose=0)[0][0]
    
    result = "REAL" if pred < 0.5 else "ALTERED"
    confidence = (1 - pred) if pred < 0.5 else pred
    
    explanations = {
        "Baseline (Original)": {
            "REAL": "The model recognizes natural, undisturbed ridge patterns characteristic of a real finger.",
            "ALTERED": "The model detected synthetic artifacts or ridge disruptions typical of an altered fingerprint."
        },
        "Rotation (90 deg)": {
            "REAL": "The system successfully identifies the biometric data despite the change in finger orientation.",
            "ALTERED": "Fraudulent markers remain clear and detectable even when the image is turned."
        },
        "Brightness (+30)": {
            "REAL": "Ridge features are robust enough to be verified even under high light or sensor exposure.",
            "ALTERED": "Alteration signatures are still visible and successfully flagged under high-exposure conditions."
        },
        "Blur (Gaussian 5x5)": {
            "REAL": "The AI extracted sufficient ridge details even with simulated lens smudge or image noise.",
            "ALTERED": "The patterns used to alter the print are distinct enough to be caught even in a blurred image."
        },
        "Inverted Colors": {
            "REAL": "The model correctly analyzes ridge structure rather than relying on simple color polarity.",
            "ALTERED": "Negative-image spoofing attempts were caught by detecting underlying ridge manipulation."
        }
    }

    reason = explanations.get(name, {}).get(result, "Classification based on neural pattern matching.")
    print(f"[{name:.<25}] Result: {result: <10} | Confidence: {confidence:.2%}")
    display_visual_result(name, img, result, confidence, reason)
    
    return pred

def start_testing(image_path):
    model = get_model()

    # USE THE NEW ROBUST IMREAD
    original = robust_imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print(f"Error: Could not load image at {image_path}")
        return

    test_scores_map = {}
    print(f"\n" + "="*70)
    print(f"   ROBUSTNESS & SECURITY REPORT: {os.path.basename(image_path)}")
    print("="*70 + "\n")
    
    test_scores_map["Baseline (Original)"] = run_inference("Baseline (Original)", original, model)
    
    rotated_90 = cv2.rotate(original, cv2.ROTATE_90_CLOCKWISE)
    test_scores_map["Rotation (90 deg)"] = run_inference("Rotation (90 deg)", rotated_90, model)
    
    bright = cv2.convertScaleAbs(original, alpha=1.2, beta=30)
    test_scores_map["Brightness (+30)"] = run_inference("Brightness (+30)", bright, model)
    
    blurred = cv2.GaussianBlur(original, (5, 5), 0)
    test_scores_map["Blur (Gaussian 5x5)"] = run_inference("Blur (Gaussian 5x5)", blurred, model)
    
    inverted = cv2.bitwise_not(original)
    test_scores_map["Inverted Colors"] = run_inference("Inverted Colors", inverted, model)
    
    # Diagnostic test
    blank = np.zeros(IMG_SIZE, dtype=np.uint8)
    run_inference("Blank (Black) Image", blank, model)
    
    print("="*70 + "\n")
    display_final_verdict(test_scores_map, os.path.basename(image_path))
    cv2.destroyAllWindows()

def select_file_and_test():
    root = tk.Tk()
    root.withdraw()
    
    print("Please select a fingerprint image to test...")
    file_path = filedialog.askopenfilename(
        title="Select Fingerprint Image for Testing",
        filetypes=[("Image Files", "*.bmp *.png *.jpg *.jpeg"), ("All Files", "*.*")]
    )
    
    root.destroy()
    if file_path:
        start_testing(file_path)
    else:
        print("No file selected. Exiting.")

if __name__ == "__main__":
    select_file_and_test()