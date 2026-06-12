import logging
import os
import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)

# ── YAMNet AudioSet class index → threat label ───────────────────────────────
UAV_INDICES = frozenset([
    0,  # Aircraft
    1,  # Fixed-wing aircraft, airplane
    2,  # Light aircraft
    3,  # Helicopter
    4,  # Aircraft engine
    5,  # Propeller, airscrew
    6,  # Drone
])

EXPLOSION_INDICES = frozenset([
    427,  # Explosion
    428,  # Gunshot, gunfire
    429,  # Artillery fire
    430,  # Burst, pop
    476,  # Boom
    431,  # Fusillade
])

YAMNET_SR = 16_000  # YAMNet очікує 16 kHz
EDGE_SR = 44_100  # Агент надсилає 44.1 kHz
MODEL_VERSION = "yamnet-tflite-v1"

# ── Resampling constants ─────────────────────────────────────────────────────
_g = gcd(YAMNET_SR, EDGE_SR)
_UP = YAMNET_SR // _g  # 160
_DOWN = EDGE_SR // _g  # 441


def _resample(samples_i16: np.ndarray) -> np.ndarray:
    """i16 @ 44.1 kHz  →  float32 [-1.0, 1.0] @ 16 kHz"""
    f32 = samples_i16.astype(np.float32) / 32768.0
    return resample_poly(f32, _UP, _DOWN)


class _YAMNetModel:
    def __init__(self, model_path: str):
        # Використовуємо вбудований tflite інтерпретатор з tensorflow
        self._interp = tf.lite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()

        in_details = self._interp.get_input_details()
        out_details = self._interp.get_output_details()

        self._input_idx = in_details[0]["index"]

        # YAMNet повертає 3 масиви: scores, embeddings, spectrogram
        # Нам потрібен тільки перший (scores)
        self._scores_idx = out_details[0]["index"]

    def predict(self, waveform_f32: np.ndarray) -> np.ndarray:
        self._interp.resize_tensor_input(self._input_idx, waveform_f32.shape)
        self._interp.allocate_tensors()
        self._interp.set_tensor(self._input_idx, waveform_f32)
        self._interp.invoke()
        scores = self._interp.get_tensor(self._scores_idx)  # [patches, 521]
        return scores.mean(axis=0)  # [521]


_model: _YAMNetModel | None = None


def load(model_path: str) -> None:
    global _model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YAMNet model not found: {model_path}")
    _model = _YAMNetModel(model_path)
    logger.info("YAMNet classifier ready (%s)", model_path)


def classify_scores(scores: np.ndarray) -> tuple[str, float]:
    """Приймає 521 ймовірність і повертає фінальний клас та впевненість"""
    uav_prob = float(np.max([scores[i] for i in UAV_INDICES]))
    exp_prob = float(np.max([scores[i] for i in EXPLOSION_INDICES]))

    if exp_prob > 0.40:
        return "EXPLOSION", exp_prob
    elif uav_prob > 0.30:
        return "UAV", uav_prob

    # Якщо нічого не перевищило поріг — це безпечний фон
    background_prob = 1.0 - max(uav_prob, exp_prob)
    return "BACKGROUND", background_prob


def classify(raw_audio_i16: list[int]) -> tuple[str, float, str]:
    if _model is None:
        raise RuntimeError("YAMNet model is not loaded.")

    samples = np.array(raw_audio_i16, dtype=np.int16)
    waveform = _resample(samples)

    mean_scores = _model.predict(waveform)
    threat, confidence = classify_scores(mean_scores)

    return threat, confidence, MODEL_VERSION
