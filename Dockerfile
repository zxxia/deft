# FROM nvidia/cuda:11.8.0-base-ubuntu20.04 as base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base

# Stop asking Geographic area in docker build
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y python3 python3-pip \
# OpenCV dependencies
# libsm6 libxext6
libgl1-mesa-glx libglib2.0-0 \
# boost dependency
libboost-all-dev

COPY ./requirements.txt ./requirements.txt
# Install python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Copy source files
COPY ./src/ ./src

# Compile hook library
WORKDIR /src/hook
RUN make

WORKDIR /
CMD ["python3", "src/dummy_server.py"]
