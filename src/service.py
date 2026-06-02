import logging
from classifier.v1 import classifier_pb2
from classifier.v1 import classifier_pb2_grpc
from config import Config

logger = logging.getLogger(__name__)


class AudioClassifierServicer(classifier_pb2_grpc.AudioClassifierServicer):
    def Classify(self, request, context):
        peak_db = request.peak_db
        fft_bins = request.fft_bins

        threat = "BACKGROUND"
        confidence = 0.99

        if peak_db > Config.EXPLOSION_THRESHOLD_DB:
            threat = "EXPLOSION"
            confidence = 0.95

        elif peak_db > Config.UAV_THRESHOLD_DB and len(fft_bins) > 0:
            low_freq_energy = sum(fft_bins[:10])
            high_freq_energy = sum(fft_bins[10:])

            if low_freq_energy > high_freq_energy * Config.UAV_FREQ_MULTIPLIER:
                threat = "UAV"
                confidence = 0.88
            else:
                threat = "BACKGROUND"
                confidence = 0.75

        logger.info(f"Classified Sensor: {request.sensor_id} | Peak DB: {peak_db:.2f} | Result: {threat}")

        return classifier_pb2.ClassificationResponse(
            threat_type=threat,
            confidence=confidence,
            model_ver=Config.MODEL_VERSION
        )
