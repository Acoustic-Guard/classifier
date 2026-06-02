import logging
import numpy as np
import joblib
import os

from classifier.v1 import classifier_pb2
from classifier.v1 import classifier_pb2_grpc
from config import Config

logger = logging.getLogger(__name__)

try:
    if os.path.exists(Config.MODEL_PATH):
        ml_model = joblib.load(Config.MODEL_PATH)
        logger.info(f"Loaded ML model from {Config.MODEL_PATH}")
    else:
        ml_model = None
        logger.warning(f"ML model not found at {Config.MODEL_PATH}. Waiting for data.")
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")
    ml_model = None


class AudioClassifierServicer(classifier_pb2_grpc.AudioClassifierServicer):
    def Classify(self, request, context):
        peak_db = request.peak_db
        fft_bins = request.fft_bins

        if peak_db > Config.EXPLOSION_THRESHOLD_DB:
            threat = "EXPLOSION"
            confidence = 0.95

        elif ml_model and len(fft_bins) > 0:
            features = np.array(list(fft_bins)[:100])

            if len(features) < 100:
                features = np.pad(features, (0, 100 - len(features)))

            features = features.reshape(1, -1)

            probs = ml_model.predict_proba(features)[0]
            max_idx = np.argmax(probs)

            threat = ml_model.classes_[max_idx]
            confidence = float(probs[max_idx])

        else:
            threat = "BACKGROUND"
            confidence = 0.99

        logger.info(f"Sensor: {request.sensor_id} | Peak DB: {peak_db:.2f} | ML Result: {threat} ({confidence:.2f})")

        return classifier_pb2.ClassificationResponse(
            threat_type=threat,
            confidence=confidence,
            model_ver=Config.MODEL_VERSION
        )
