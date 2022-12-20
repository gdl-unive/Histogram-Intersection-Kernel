FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
WORKDIR /root
RUN apt update && apt install -y curl tar xz-utils gcc cmake git \
    build-essential \
    gdb \
    lcov \
    pkg-config \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libgdbm-compat-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    lzma \
    lzma-dev \
    tk-dev \
    uuid-dev \
    zlib1g-dev
RUN curl -O https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tar.xz && \
    tar -xf Python-3.8.12.tar.xz && \
    cd Python-3.* && \
    ./configure && \
    make install
RUN pip3 install autopep8
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.0 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install torch-scatter==2.0.5 torch-sparse==0.6.9 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==1.6.3 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
COPY requirements2.txt /tmp/requirements2.txt
RUN pip3 install -r /tmp/requirements2.txt
