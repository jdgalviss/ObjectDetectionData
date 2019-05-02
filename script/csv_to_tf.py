import os
import contextlib2
import subprocess
#from object_detection.dataset_tools import tf_record_creation_util
bash_command = "ls ../CSV"
process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
files = output
files = str(output).replace("b","")
files = str(files).replace("'","")
files = files.split('\\n')
files.pop()

print("Creating TFRecords...")
for fname in files:
    name = fname.split('.')[0]
    input_path = '../CSV/' + fname
    output_path = '../data/TFRecords/' + name +'.record'
    images = '../data/images' 
    bash_command = "python ../csv_a_tf.py --csv_input=" + input_path + " --output_path=" + output_path + " --images="+images
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    print(error)
print("TFRecords created")