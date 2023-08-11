FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG UNAME=pip
ARG UID=1000
ARG GID=1000
ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update
RUN apt-get -y --fix-missing install git python3 python3-pip build-essential ffmpeg libsm6 libxext6 cmake

RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

USER $UNAME

COPY requirements.txt .
RUN pip3 install --upgrade pip 
RUN pip3 install --no-cache-dir yapf
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 torch torchvision
RUN pip3 install --no-cache-dir  -r requirements.txt