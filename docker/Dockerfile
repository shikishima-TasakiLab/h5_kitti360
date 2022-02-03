FROM ubuntu:20.04

SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        git \
        cmake \
        python3-pip \
        python3-dev \
        python3-numpy-dev \
        libboost-python-dev \
        libboost-numpy-dev \
        libpcl-dev \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_numpy3*.so /usr/lib/x86_64-linux-gnu/libboost_numpy3.so ; \
    fi; \
    if [[ ! -f /usr/lib/x86_64-linux-gnu/libboost_python3.so ]]; then \
        ln -s /usr/lib/x86_64-linux-gnu/libboost_python3*.so /usr/lib/x86_64-linux-gnu/libboost_python3.so ; \
    fi

RUN if [[ -f /opt/conda/bin/activate ]]; then \
        source /opt/conda/bin/activate && \
        conda install -c conda-forge scikit-build ninja h5py==2.10.0; \
    else \
        pip3 install scikit-build ninja h5py==2.10.0; \
    fi; \
    pip3 install git+https://github.com/shikishima-TasakiLab/pointsmap-python

RUN pip3 install git+https://github.com/shikishima-TasakiLab/h5datacreator
RUN pip3 install git+https://github.com/shikishima-TasakiLab/h5_kitti360
