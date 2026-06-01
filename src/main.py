import grpc
from concurrent import futures
import logging

from pb.classifier.v1 import classifier_pb2_grpc
from service import AudioClassifierServicer


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    classifier_pb2_grpc.add_AudioClassifierServicer_to_server(AudioClassifierServicer(), server)

    server.add_insecure_port('[::]:50051')

    logging.info("Starting gRPC server on port 50051...")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    serve()