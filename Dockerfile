# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.9    (apt)
# pytorch       1.9 (pip)
# ==================================================================

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim \
        fish \
        libssl-dev \
        libsparsehash-dev \
	libblas-dev \
	liblapack-dev \
	libhdf5-dev \
        && \
# ==================================================================
# python
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.9 \
        python3.9-dev \
	python3.9-distutils \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.9 ~/get-pip.py && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        matplotlib \
        Cython \
        && \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL torch torchvision && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        shapely fire pybind11 tensorboardX protobuf \
        scikit-image numba pillow

WORKDIR /app
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
RUN tar xzvf boost_1_76_0.tar.gz
RUN cp -r ./boost_1_76_0/boost /usr/include
RUN rm -rf ./boost_1_76_0
RUN rm -rf ./boost_1_76_0.tar.gz
RUN git clone https://github.com/traveller59/spconv.git --depth 10 --recursive

RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1.tar.gz
RUN tar zxvf cmake-3.21.1.tar.gz
RUN cd cmake-3.21.1/ && \
	./bootstrap && \
	make && make install
ENV PATH /app/cmake-3.21.1/bin:$PATH
RUN rm -rf ./cmake-3.21.1.tar.gz

# ==================================================================
# Env for cuda compiler. See https://github.com/pytorch/extension-cpp/issues/71
# How to find your GPU's CC: https://developer.nvidia.com/cuda-gpus
# ==================================================================
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"
ENV SPCONV_FORCE_BUILD_CUDA 1
RUN cd ./spconv && python setup.py bdist_wheel && pip install ./dist/spconv*.whl
ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice

RUN mkdir -p /app/second.pytorch
ENV PYTHONPATH=/app/second.pytorch
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
	apt-get update && \
	$APT_INSTALL libgl1-mesa-dev
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
	$PIP_INSTALL tqdm opencv-python seaborn psutil tensorboard
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
	$PIP_INSTALL Flask flask-cors

COPY torchplus/ /app/second.pytorch/torchplus/
COPY second/ /app/second.pytorch/second/

VOLUME ["/app/data"]
VOLUME ["/app/model"]
WORKDIR /app/second.pytorch/second

