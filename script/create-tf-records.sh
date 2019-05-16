#!/bin/bash
#Object detection tensorflow scripts
cd /usr/src/app
mkdir trash
cd trash
wget https://github.com/puigalex/deteccion_objetos/archive/master.zip
unzip master.zip
cd deteccion_objetos-master/
cp -r object_detection/ ../../
cd ../../
rm -rf trash/

if [ -d /usr/src/app/data/TFRecords ]; then
    rm -r /usr/src/app/data/TFRecords/*
fi

if [ ! -d /usr/src/app/data/TFRecords ]; then
    mkdir /usr/src/app/data/TFRecords
fi

cd script
python json_to_csv.py
python csv_to_tf.py