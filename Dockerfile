FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         libturbojpeg \
         libgl1-mesa-glx &&\
     rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH
RUN conda install pytorch torchvision cuda100 -c pytorch

RUN conda install -c conda-forge opencv \
    jupyter \
    pandas \
    matplotlib \
    tqdm \
    scikit-learn 

RUN pip install \
    pytorch-argus==0.0.8 \
    cnn-finetune==0.5.1 

RUN apt-get update &&\
    apt-get install -y libsnappy-dev &&\
    pip install parquet==1.2 \
    pyarrow==0.12.0

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex && python setup.py install

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
