import os
import glob
import numpy as np
import joblib
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def extract_fft_features_from_segments(file_path, num_bins=100, target_sr=44100):
    sr, y = wavfile.read(file_path)

    if sr != target_sr:
        print(f"Warning: Sample rate of {file_path} is {sr}Hz. Expected {target_sr}Hz. Frequencies may shift.")

    if len(y.shape) > 1:
        y = y.mean(axis=1)

    segment_size = target_sr
    features_list = []

    if len(y) < segment_size:
        padded = np.pad(y, (0, segment_size - len(y)))
        fft_complex = np.fft.rfft(padded)
        fft_spectrum = np.abs(fft_complex)
        features = fft_spectrum[:num_bins]
        if len(features) < num_bins:
            features = np.pad(features, (0, num_bins - len(features)))
        features_list.append(features)
    else:
        for offset in range(0, len(y) - segment_size + 1, segment_size):
            segment = y[offset:offset + segment_size]
            fft_complex = np.fft.rfft(segment)
            fft_spectrum = np.abs(fft_complex)
            features = fft_spectrum[:num_bins]
            if len(features) < num_bins:
                features = np.pad(features, (0, num_bins - len(features)))
            features_list.append(features)

    return features_list


def load_dataset(base_path="dataset"):
    X = []
    y = []

    classes = ["BACKGROUND", "UAV", "EXPLOSION"]

    for label in classes:
        folder_path = os.path.join(base_path, label)

        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} not found. Skipping.")
            continue

        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"Processing directory for class {label}...")

        class_segments_count = 0
        for file_path in wav_files:
            try:
                features_list = extract_fft_features_from_segments(file_path)
                for features in features_list:
                    X.append(features)
                    y.append(label)
                    class_segments_count += 1
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        print(f"Generated {class_segments_count} total 1-second segments for class {label}")

    return np.array(X), np.array(y)


def train_model():
    print("1. Loading and slicing data from dataset/ directory...")
    X, y = load_dataset("dataset")

    if len(X) == 0:
        print("Error: No data available for training! Please add .wav files to dataset subdirectories.")
        return

    print(f"\nSuccessfully processed {len(X)} total audio records.")

    print("2. Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    print("\n4. Evaluating model performance (Classification Report):")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    model_path = "models/random_forest_v1.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel successfully trained and saved to {model_path}")


if __name__ == "__main__":
    train_model()
