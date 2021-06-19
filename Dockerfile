FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

ENV PYTHONUNBUFFERED TRUE

# FASTAI
RUN apt-get update && apt-get install -y software-properties-common rsync
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev graphviz && apt-get update
RUN pip install albumentations \
    catalyst \
    captum \
    "fastprogress>=0.1.22" \
    graphviz \
    kornia \
    matplotlib \
    "nbconvert<6"\
    neptune-cli \
    opencv-python \
    pandas \
    pillow \
    pyarrow \
    pydicom \
    pyyaml \
    scikit-learn \
    scikit-image \
    scipy \
    "sentencepiece<0.1.90" \
    spacy \
    tensorboard \
    wandb

# TORCHSERVE
RUN DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    openjdk-11-jdk \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# FASTAI
RUN git clone https://github.com/fastai/fastai.git && git clone https://github.com/fastai/fastcore.git
RUN /bin/bash -c "cd fastai && git checkout 2.0.18 && pip install . && cd ../fastcore && git checkout 1.1.0 && pip install ."

# TORCHSERVE
RUN git clone https://github.com/pytorch/serve.git
RUN pip install ./serve/
RUN pip install captum

COPY ./deployment/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY ./deployment/config.properties /home/model-server/config.properties

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
