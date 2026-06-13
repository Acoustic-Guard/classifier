import logging
import os
import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly
from math import gcd
from config import Config

logger = logging.getLogger(__name__)

# ── YAMNet AudioSet class index → threat label ───────────────────────────────
EXPLOSION_INDICES = frozenset([
    395,  # Thunder
    420,  # Gunshot, gunfire
    426,  # Gunshot, gunfire
    427,  # Explosion
    428,  # Gunshot, gunfire
    429,  # Artillery fire
    430,  # Burst, pop
    431,  # Fusillade
    432,  # Burst, pop
    476,  # Boom
])

UAV_INDICES = frozenset([
    132,  # Truck
    137,  # Moped
    351,  # Chainsaw
    352,  # Power tool
    354,  # Drill
    355,  # Power saw
    382,  # Lawn mower
    384,  # Trimmer
])

SIREN_INDICES = frozenset([
    319,  # Siren
    320,  # Civil defense siren
    321,  # Air raid siren
    322,  # Alarm
    323,  # Fire alarm
    390,  # Siren
])

TRUCK_INDICES = frozenset([
    134,  # Bus
    135,  # Car
    312,  # Horn
    313,  # Horn
    393,  # Horn
])

GENERATOR_INDICES = frozenset([
    130,  # Engine
    131,  # Motor
    133,  # Machinery
    294,  # Generator
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
        self._input_shape = in_details[0]["shape"]

        # YAMNet повертає 3 масиви: scores, embeddings, spectrogram
        # Нам потрібен тільки перший (scores)
        self._scores_idx = out_details[0]["index"]

    def predict(self, waveform_f32: np.ndarray) -> np.ndarray:
        # Dynamically get expected input size from interpreter
        expected_size = self._input_shape[0]

        # Shape enforcement: pad or slice to match expected size
        current_size = len(waveform_f32)
        if current_size < expected_size:
            # Pad with zeros (silence) if too short
            waveform_f32 = np.pad(waveform_f32, (0, expected_size - current_size), 'constant')
        elif current_size > expected_size:
            # Slice down to exact size if too long
            waveform_f32 = waveform_f32[:expected_size]

        # Ensure data type remains float32
        waveform_f32 = waveform_f32.astype(np.float32)

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
    # Log raw prediction for debugging
    top_index = int(np.argmax(scores))
    max_prob = float(scores[top_index])
    logger.info(f"YAMNet raw top index predicted: {top_index} with prob: {max_prob:.2f}")

    # Route based on top predicted index
    if top_index in EXPLOSION_INDICES and max_prob > Config.YAMNET_EXPLOSION_CONFIDENCE:
        return "EXPLOSION", max_prob
    elif top_index in UAV_INDICES and max_prob > Config.YAMNET_UAV_CONFIDENCE:
        return "UAV", max_prob
    elif top_index in SIREN_INDICES and max_prob > Config.YAMNET_SIREN_CONFIDENCE:
        return "SIREN", max_prob
    elif top_index in TRUCK_INDICES and max_prob > Config.YAMNET_TRUCK_CONFIDENCE:
        return "TRUCK", max_prob
    elif top_index in GENERATOR_INDICES and max_prob > Config.YAMNET_GENERATOR_CONFIDENCE:
        return "GENERATOR", max_prob
    else:
        return "BACKGROUND", max_prob


def classify(raw_audio_i16: list[int]) -> tuple[str, float, str]:
    if _model is None:
        raise RuntimeError("YAMNet model is not loaded.")

    if len(raw_audio_i16) == 0:
        logger.warning("Empty raw_audio received, returning BACKGROUND")
        return "BACKGROUND", 0.99, MODEL_VERSION

    samples = np.array(raw_audio_i16, dtype=np.int16)
    waveform = _resample(samples)

    mean_scores = _model.predict(waveform)
    threat, confidence = classify_scores(mean_scores)

    return threat, confidence, MODEL_VERSION
