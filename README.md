# Classifier

## Overview
The `classifier` is a Python-based gRPC service responsible for real-time acoustic threat classification. It is designed to process audio frames and accurately identify various anomalies, specifically:
- UAV (Drones)
- Explosion
- Siren
- Truck
- Generator

## ML Pipeline
This service utilizes a dual Machine Learning pipeline to ensure robust classification:
1. **YAMNet:** Used primarily for embeddings and feature extraction from the incoming audio frames.
2. **Random Forest:** Consumes the features extracted by YAMNet to perform the final classification into the defined threat categories.

## Environment Variables
The service can be configured using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CLASSIFIER_PORT` | The port on which the gRPC service listens. | `3232` |
| `LOG_LEVEL` | Logging level for the application. | `INFO` |
| `MODEL_PATH` | Path to the Random Forest model file. | `models/random_forest_v1.joblib` |
| `MODEL_VERSION` | Identifier for the loaded model version. | `rf-v2.0` |
| `YAMNET_MODEL_PATH` | Path to the YAMNet TensorFlow Lite model. | `models/yamnet.tflite` |
| `EXPLOSION_THRESHOLD_DB` | Decibel threshold for detecting potential explosions. | `-10.0` |
| `UAV_THRESHOLD_DB` | Decibel threshold for detecting potential UAVs. | `-30.0` |
| `YAMNET_EXPLOSION_CONFIDENCE` | Minimum confidence score for explosion classification. | `0.10` |
| `YAMNET_UAV_CONFIDENCE` | Minimum confidence score for UAV classification. | `0.10` |
| `YAMNET_SIREN_CONFIDENCE` | Minimum confidence score for siren classification. | `0.10` |
| `YAMNET_TRUCK_CONFIDENCE` | Minimum confidence score for truck classification. | `0.10` |
| `YAMNET_GENERATOR_CONFIDENCE` | Minimum confidence score for generator classification. | `0.10` |

## Setup & Running

### Local Development
1. Navigate to the `classifier` directory.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If necessary, generate the gRPC stubs from your `.proto` files (assuming standard `grpc_tools.protoc` usage).
4. Run the service:
   ```bash
   python src/main.py
   ```

### Running via Docker
The application is fully containerized. To run it using Docker:
```bash
docker build -t acoustic_guard_classifier .
docker run -d -p 3232:3232 --env-file .env acoustic_guard_classifier
```
