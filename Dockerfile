FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    unzip \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    netbase \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

#Make directories
RUN /bin/bash -c "mkdir -p /usr/src/app/{CSV,output,train,script,annotations}"

# Copy scripts used to prepare data
COPY script /usr/src/app/script
COPY annotations /usr/src/app/annotations
COPY csv_a_tf.py /usr/src/app/

# Ready to Download object_detection to use object_detection utils/Download images / 
# Parse annotation files (json from scalabel) to csv / Create tf records
# CMD ["bash", "/usr/src/app/create-tf-records.sh"]

# Install object detection training dependencies

COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    python3-setuptools \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/* \
    && pip install -r requirements.txt \
    && pip install numpy
RUN pip install pycocotools

# Clone object detection tensorflow/models repository
WORKDIR /usr/src/app
RUN /bin/bash -c "git clone https://github.com/tensorflow/models.git"

# COCO API Installation
WORKDIR /usr/src/app
RUN /bin/bash -c "git clone https://github.com/cocodataset/cocoapi.git; cd cocoapi/PythonAPI; make; cp -r pycocotools /usr/src/app/models/research/"

# Protobuf compilation
WORKDIR /usr/src/app/models/research
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN /bin/bash -c "./bin/protoc object_detection/protos/*.proto --python_out=."
ENV PYTHONPATH=$PYTHONPATH:/usr/src/app/models/research:/usr/src/app/models/research/slim
ENV PIPELINE_CONFIG_PATH=/usr/src/app/data/models/ssd_mobilenet_v2/ssd_mobilenet_v2_coco.config
ENV MODEL_DIR=/usr/src/app/data/train_data
ENV NUM_TRAIN_STEPS=4000
ENV SAMPLE_1_OF_N_EVAL_EXAMPLES=1
