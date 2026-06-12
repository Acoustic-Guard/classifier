import os
import glob
import numpy as np
import joblib
from scipy.io import wavfile
from scipy.signal import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

TARGET_SR = 44100
NUM_BINS = 100
SEGMENT_SIZE = TARGET_SR  # 1 second


def extract_features(segment: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract a richer feature vector from a 1-second audio segment.

    Features:
      - 100 FFT magnitude bins (normalised)
      - RMS energy (scalar)                 → helps distinguish silence
      - Zero-crossing rate (scalar)         → helps distinguish noise from tonal UAV
      - Spectral centroid (scalar)          → UAVs tend to have low-mid centroid
      - Spectral flatness (scalar)          → explosions are broadband / impulsive
    Total: 104 features
    """
    if len(segment.shape) > 1:
        segment = segment.mean(axis=1)

    segment = segment.astype(np.float32)

    fft_mag = np.abs(np.fft.rfft(segment))[:NUM_BINS]
    if len(fft_mag) < NUM_BINS:
        fft_mag = np.pad(fft_mag, (0, NUM_BINS - len(fft_mag)))

    fft_max = np.max(fft_mag)
    if fft_max > 0:
        fft_norm = fft_mag / fft_max
    else:
        fft_norm = fft_mag

    rms = np.sqrt(np.mean(segment ** 2))

    rms_norm = np.clip(rms / 32768.0, 0, 1)

    zcr = np.mean(np.abs(np.diff(np.sign(segment)))) / 2.0

    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)[:NUM_BINS]
    if fft_max > 0:
        centroid = np.sum(freqs * fft_norm) / (np.sum(fft_norm) + 1e-9)
        centroid_norm = centroid / (sr / 2.0)
    else:
        centroid_norm = 0.0

    eps = 1e-9
    geometric_mean = np.exp(np.mean(np.log(fft_mag + eps)))
    arithmetic_mean = np.mean(fft_mag) + eps
    flatness = geometric_mean / arithmetic_mean

    return np.concatenate([fft_norm, [rms_norm, zcr, centroid_norm, flatness]])


def is_silent(segment: np.ndarray, threshold_rms: float = 200.0) -> bool:
    """
    Return True if the segment is essentially silence.
    threshold_rms ~ 200 out of 32768 ≈ -44 dB FS.
    Adjust if your recordings are quieter.
    """
    segment = segment.astype(np.float32)
    if len(segment.shape) > 1:
        segment = segment.mean(axis=1)
    return np.sqrt(np.mean(segment ** 2)) < threshold_rms


def augment_segment(segment: np.ndarray) -> list:
    """
    Return a list of augmented copies of *segment* (excluding the original).
    Augmentations:
      - Gaussian noise (2 levels)
      - Amplitude scale (quiet / loud)
      - Time shift (±10 %)
    """
    aug = []
    seg = segment.astype(np.float32)

    for noise_std in [0.005, 0.015]:
        noisy = seg + np.random.randn(*seg.shape).astype(np.float32) * noise_std * np.max(np.abs(seg) + 1e-9)
        aug.append(noisy)

    for scale in [0.6, 1.4]:
        aug.append(np.clip(seg * scale, -32768, 32767).astype(np.float32))

    for shift in [int(SEGMENT_SIZE * 0.1), -int(SEGMENT_SIZE * 0.1)]:
        aug.append(np.roll(seg, shift))

    return aug


def load_dataset(base_path: str = "dataset", augment_classes: list = None):
    """
    Load .wav files, slice into 1-second segments, optionally augment
    under-represented classes.

    augment_classes: list of class names to augment (e.g. ["EXPLOSION"])
    """
    if augment_classes is None:
        augment_classes = []

    X, y = [], []
    classes = ["BACKGROUND", "UAV", "EXPLOSION"]

    for label in classes:
        folder_path = os.path.join(base_path, label)
        if not os.path.exists(folder_path):
            print(f"  [WARN] Directory not found: {folder_path} – skipping.")
            continue

        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"\n  [{label}] found {len(wav_files)} files …")

        seg_count = 0
        aug_count = 0
        skipped_silent = 0

        for file_path in wav_files:
            try:
                sr, raw = wavfile.read(file_path)
            except Exception as e:
                print(f"    [ERROR] {file_path}: {e}")
                continue

            if sr != TARGET_SR:
                # Simple integer resampling (good-enough for training data)
                factor = TARGET_SR / sr
                raw = resample(raw, int(len(raw) * factor)).astype(raw.dtype)

            if len(raw.shape) > 1:
                raw = raw.mean(axis=1)

            # Slice into 1-second segments
            if len(raw) < SEGMENT_SIZE:
                raw = np.pad(raw, (0, SEGMENT_SIZE - len(raw)))
                segments = [raw]
            else:
                segments = [
                    raw[offset: offset + SEGMENT_SIZE]
                    for offset in range(0, len(raw) - SEGMENT_SIZE + 1, SEGMENT_SIZE)
                ]

            for seg in segments:
                if is_silent(seg):
                    skipped_silent += 1
                    continue

                feats = extract_features(seg, TARGET_SR)
                X.append(feats)
                y.append(label)
                seg_count += 1

                if label in augment_classes:
                    for aug_seg in augment_segment(seg):
                        if not is_silent(aug_seg):
                            X.append(extract_features(aug_seg, TARGET_SR))
                            y.append(label)
                            aug_count += 1

        print(f"    segments: {seg_count}  |  augmented: {aug_count}  |  silent skipped: {skipped_silent}")

    return np.array(X), np.array(y)


def train_model(dataset_path: str = "dataset", out_path: str = "models/random_forest_v1.joblib"):
    print("=" * 60)
    print("Step 1 – Loading dataset …")
    print("=" * 60)

    X, y = load_dataset(dataset_path, augment_classes=["EXPLOSION"])

    if len(X) == 0:
        print("[ERROR] No data found – add .wav files to dataset/<CLASS>/ directories.")
        return

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Total samples after augmentation: {len(X)}")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt}")

    print("\n" + "=" * 60)
    print("Step 2 – Train / test split …")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n" + "=" * 60)
    print("Step 3 – Training RandomForest …")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("Step 4 – Calibrating probabilities (Platt scaling) …")
    print("=" * 60)

    calibrated = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("Step 5 – Evaluation …")
    print("=" * 60)
    y_pred = calibrated.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=["BACKGROUND", "UAV", "EXPLOSION"])
    print("            BG   UAV  EXP")
    for lbl, row in zip(["BACKGROUND", "UAV       ", "EXPLOSION "], cm):
        print(f"  {lbl}  {row}")

    print("\n[Sanity] all-zeros input →", end=" ")
    z = np.zeros((1, 104))
    probs = calibrated.predict_proba(z)[0]
    classes_ = calibrated.classes_
    pred = classes_[np.argmax(probs)]
    print(f"{pred}  probs={dict(zip(classes_, probs.round(3)))}")

    print("\n" + "=" * 60)
    print("Step 6 – Saving model …")
    print("=" * 60)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    joblib.dump(calibrated, out_path)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    train_model()
