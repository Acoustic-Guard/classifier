import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    PORT = os.getenv("CLASSIFIER_PORT", "3232")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_v2.joblib")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "rf-v2.0")

    YAMNET_MODEL_PATH = os.getenv("YAMNET_MODEL_PATH", "models/yamnet.tflite")
