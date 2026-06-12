import logging
import numpy as np
import joblib
import os

from classifier.v1 import classifier_pb2
from classifier.v1 import classifier_pb2_grpc
from config import Config
import yamnet_classifier

logger = logging.getLogger(__name__)

try:
    if os.path.exists(Config.MODEL_PATH):
        _rf_model = joblib.load(Config.MODEL_PATH)
        logger.info("Loaded RF model from %s", Config.MODEL_PATH)
    else:
        _rf_model = None
        logger.warning("RF model not found at %s", Config.MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load RF model: %s", exc)
    _rf_model = None

try:
    yamnet_classifier.load(Config.YAMNET_MODEL_PATH)
except Exception as exc:
    logger.error("Failed to load YAMNet model: %s", exc)


class AudioClassifierServicer(classifier_pb2_grpc.AudioClassifierServicer):

    def Classify(self, request, context):
        has_raw = len(request.raw_audio) > 0
        has_fft = len(request.fft_bins) > 0

        if has_raw:
            threat, confidence, model_ver = self._classify_raw(request)
        elif has_fft:
            threat, confidence, model_ver = self._classify_fft(request)
        else:
            logger.warning("Sensor %s sent empty payload", request.sensor_id)
            threat, confidence, model_ver = "BACKGROUND", 0.99, "fallback-0.0"

        logger.info("Sensor: %s | Engine: %s | Result: %s (%.2f)",
                    request.sensor_id, model_ver, threat, confidence)

        return classifier_pb2.ClassificationResponse(
            threat_type=threat,
            confidence=confidence,
            model_ver=model_ver
        )

    def _classify_raw(self, request):
        try:
            return yamnet_classifier.classify(list(request.raw_audio))
        except Exception as e:
            logger.error(f"YAMNet inference failed: {e}")
            return "BACKGROUND", 0.99, "error-yamnet"

    def _classify_fft(self, request):
        if not _rf_model:
            return "BACKGROUND", 0.99, "error-rf-missing"

        try:
            features = np.array(list(request.fft_bins)[:100], dtype=float)

            if len(features) < 100:
                features = np.pad(features, (0, 100 - len(features)))

            max_val = np.max(features)
            if max_val > 0:
                features = features / max_val

            features = features.reshape(1, -1)

            probs = _rf_model.predict_proba(features)[0]
            max_idx = np.argmax(probs)

            threat = _rf_model.classes_[max_idx]
            confidence = float(probs[max_idx])

            return threat, confidence, Config.MODEL_VERSION

        except Exception as e:
            logger.error(f"Custom model inference failed: {e}")
            return "BACKGROUND", 0.99, "error-rf"
