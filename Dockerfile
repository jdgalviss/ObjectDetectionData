FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN apt-get update

RUN apt-get install -y \
    git \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    netbase

#Make directories
RUN mkdir -p /usr/src/app/{images,CSV,output,TFRecords,train} 

# Copy scripts used to prepare data
COPY script /usr/src/app/
COPY annotations /usr/src/app/
COPY config /usr/src/app/
COPY models /usr/src/app/
COPY csv_a_tf.py /usr/src/app/

# Download object_detection to use object_detection utils/Download images / 
# Parse annotation files (json from scalabel) to csv / Create tf records

CMD ["bash", "/usr/src/appcreate-tf-records.sh"]