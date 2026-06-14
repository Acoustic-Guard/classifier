import grpc
import signal
import logging
import sys
from concurrent import futures

from classifier.v1 import classifier_pb2_grpc
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
    
    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        server.stop(grace=None)
        logger.info("gRPC server stopped gracefully")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    server.wait_for_termination()


if __name__ == '__main__':
    serve()
