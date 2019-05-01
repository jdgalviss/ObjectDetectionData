
cd /usr/src/app
if [ -d "mkdir /usr/src/app/data/images" ]; then
    rm -rf /usr/src/app/data/images
fi
mkdir /usr/src/app/data/images
cd script
python get_images.py