import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    PORT = os.getenv("CLASSIFIER_PORT", "3232")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_v1.joblib")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "rf-v2.0")

    YAMNET_MODEL_PATH = os.getenv("YAMNET_MODEL_PATH", "models/yamnet.tflite")

    EXPLOSION_THRESHOLD_DB = float(os.getenv("EXPLOSION_THRESHOLD_DB", "-10.0"))
    UAV_THRESHOLD_DB = float(os.getenv("UAV_THRESHOLD_DB", "-30.0"))

    YAMNET_EXPLOSION_CONFIDENCE = float(os.getenv("YAMNET_EXPLOSION_CONFIDENCE", "0.15"))
    YAMNET_UAV_CONFIDENCE = float(os.getenv("YAMNET_UAV_CONFIDENCE", "0.15"))
    YAMNET_SIREN_CONFIDENCE = float(os.getenv("YAMNET_SIREN_CONFIDENCE", "0.15"))
    YAMNET_TRUCK_CONFIDENCE = float(os.getenv("YAMNET_TRUCK_CONFIDENCE", "0.15"))
    YAMNET_GENERATOR_CONFIDENCE = float(os.getenv("YAMNET_GENERATOR_CONFIDENCE", "0.15"))