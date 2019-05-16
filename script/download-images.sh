#!/bin/bash
cd /usr/src/app
if [ -d /usr/src/app/data/images ]; then
    rm -r /usr/src/app/data/images/*
fi
if [ ! -d /usr/src/app/data/images ]; then
    mkdir /usr/src/app/data/images
fi

cd script
python get_images.py