#Object detection tensorflow scripts
cd ..
mkdir trash
cd trash
wget https://github.com/puigalex/deteccion_objetos/archive/master.zip
unzip master.zip
cd deteccion_objetos-master/
cp -r object_detection/ ../../
cd ../../
rm -rf trash/