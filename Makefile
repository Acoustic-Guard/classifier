PROTO_DIR = ../proto
SRC_DIR = src
PB_DIR = $(SRC_DIR)/pb

.PHONY: install proto run clean

install:
	pip install -r requirements.txt

install-up:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

proto:
	mkdir -p $(PB_DIR)
	python3 -m grpc_tools.protoc -I$(PROTO_DIR) --python_out=$(PB_DIR) --grpc_python_out=$(PB_DIR) $(PROTO_DIR)/classifier/v1/classifier.proto

run:
	PYTHONPATH=$(SRC_DIR):$(PB_DIR) python3 $(SRC_DIR)/main.py

clean:
	rm -rf $(PB_DIR)
	rm -rf $(SRC_DIR)/__pycache__