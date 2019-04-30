
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

cd scripts
python3 get_images.py
python3 json_to_csv.py
python3 csv_to_tf.py