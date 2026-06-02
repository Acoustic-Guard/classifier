import grpc
from concurrent import futures
import logging
import sys

from pb.classifier.v1 import classifier_pb2_grpc
from service import AudioClassifierServicer
from config import Config


def setup_logging():
    numeric_level = getattr(logging, Config.LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def serve():
    setup_logging()
    logger = logging.getLogger(__name__)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    classifier_pb2_grpc.add_AudioClassifierServicer_to_server(AudioClassifierServicer(), server)

    listen_addr = f'[::]:{Config.PORT}'
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting gRPC Audio Classifier on {listen_addr}...")
    logger.info(f"Config: EXPLOSION_DB > {Config.EXPLOSION_THRESHOLD_DB}, UAV_DB > {Config.UAV_THRESHOLD_DB}")

    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
