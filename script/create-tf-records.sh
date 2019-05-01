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

if [ -d "mkdir /usr/src/app/data/TFRecords" ]; then
    rm -rf /usr/src/app/data/TFRecords
fi
mkdir /usr/src/app/data/TFRecords
cd script
python json_to_csv.py
python csv_to_tf.py