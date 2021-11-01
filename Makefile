IMAGE_NAME := second.pytorch
MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
KITTI_DATASET_ROOT :=

.PHONY: build
build:
	docker build \
		-t $(IMAGE_NAME):latest .

.PHONY: prepare
prepare:
ifndef KITTI_DATASET_ROOT
	echo "argument KITTI_DATASET_ROOT is not defined"
	exit 1
endif
	mkdir -p $(KITTI_DATASET_ROOT)/training/velodyne_reduced && \
	mkdir -p $(KITTI_DATASET_ROOT)/testing/velodyne_reduced && \
	docker run --rm -it --gpus all \
		-v /hdd/kitti:/app/data \
		-v $(MAKEFILE_DIR)/model:/app/model \
		$(IMAGE_NAME):latest \
		python create_data.py kitti_data_prep --root_path=/app/data
       	
.PHONY: prepare-sing
prepare-sing:
ifndef KITTI_DATASET_ROOT
	echo "argument KITTI_DATASET_ROOT is not defined"
	exit 1
endif
	mkdir -p $(KITTI_DATASET_ROOT)/training/velodyne_reduced && \
	mkdir -p $(KITTI_DATASET_ROOT)/testing/velodyne_reduced && \
	singularity exec --nv \
		--pwd $(MAKEFILE_DIR)/second \
		./singularity/second-pytorch.simg \
		python create_data.py kitti_data_prep --root_path=$(KITTI_DATASET_ROOT)

CONF := pointpillars/car/xyres_16.config
EXP := $(subst .config,,$(subst /,-,$(CONF)))
.PHONY: train
train:
ifndef KITTI_DATASET_ROOT
	echo "argument KITTI_DATASET_ROOT is not defined"
	exit 1
endif
	echo CONF:$(CONF), EXP:$(EXP)
	docker run --rm -it --gpus all \
		-v /hdd/kitti:/app/data \
		-v $(MAKEFILE_DIR)/model:/app/model \
		$(IMAGE_NAME):latest \
		python ./pytorch/train.py train \
			--config_path=./configs/$(CONF) \
			--model_dir=/app/model/$(EXP)
	docker run --rm -it \
		-v $(MAKEFILE_DIR)/model:/app/model \
		$(IMAGE_NAME):latest \
		/bin/sh -c "echo `git rev-parse HEAD` > /app/model/$(EXP)/commit-hash.txt"

.PHONY: train-sing
train-sing:
	echo "conf: $(CONF) exp: $(EXP)"
ifndef KITTI_DATASET_ROOT
	echo "argument KITTI_DATASET_ROOT is not defined"
	exit 1
endif
	singularity exec --nv \
		--pwd $(MAKEFILE_DIR)/second \
		--bind $(KITTI_DATASET_ROOT):/app/data \
		./singularity/second-pytorch.simg \
		python ./pytorch/train.py train \
			--config_path=./configs/$(CONF) \
			--model_dir=$(MAKEFILE_DIR)/model/$(EXP)

.PHONY: eval
eval:
ifndef KITTI_DATASET_ROOT
	echo "argument KITTI_DATASET_ROOT is not defined"
	exit 1
endif
	docker run --rm -it --gpus all \
		-v /hdd/kitti:/app/data \
		-v $(MAKEFILE_DIR)/model:/app/model \
		$(IMAGE_NAME):latest \
		python ./pytorch/train.py evaluate \
			--config_path=./configs/$(CONF) \
			--model_dir=/app/model/$(EXP) \
			--measure_time=True \
			--batch_size=1
	docker run --rm -it \
		-v $(MAKEFILE_DIR)/model:/app/model \
		$(IMAGE_NAME):latest \
		/bin/sh -c "echo `git rev-parse HEAD` > /app/model/$(EXP)/eval_results/commit-hash.txt"
.PHONY: board
board:
	docker run --rm -it \
		-v $(MAKEFILE_DIR)/model:/app/model \
		-p 6006:6006 \
		$(IMAGE_NAME):latest \
		tensorboard --logdir /app/model --port 6006 --bind_all

BACKEND_PORT := 16666
.PHONY: viewer
viewer:
	docker run --rm -it --gpus all \
		-v /hdd/kitti:/app/data \
		-v $(MAKEFILE_DIR)/model:/app/model \
		--publish=$(BACKEND_PORT):$(BACKEND_PORT) \
		$(IMAGE_NAME):latest \
		python ./kittiviewer/backend/main.py main --port=$(BACKEND_PORT)
