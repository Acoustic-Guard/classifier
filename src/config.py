import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    PORT = os.getenv("CLASSIFIER_PORT", "3232")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    EXPLOSION_THRESHOLD_DB = float(os.getenv("EXPLOSION_THRESHOLD_DB", "-10.0"))
    UAV_THRESHOLD_DB = float(os.getenv("UAV_THRESHOLD_DB", "-30.0"))
    UAV_FREQ_MULTIPLIER = float(os.getenv("UAV_FREQ_MULTIPLIER", "1.5"))

    MODEL_PATH = os.getenv("MODEL_PATH", "models/random_forest_v1.joblib")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "rf-v1.0")
