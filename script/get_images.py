import json
import urllib.request
import os
import subprocess
#Getting annotations file names
bash_command = "ls ../annotations"
process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
#print(str(output).replace("b",""))
files = str(output).replace("b","")
files = str(files).replace("'","")
files = files.split('\\n')
files.pop()
path = '../annotations/'
savePath = '../images/'
print("Downloading images...")
for fname in files:
    with open(path + fname) as json_file:
        
        data = json.load(json_file)
        
        for image in data:
            url = image['url']
            name = url.split('/')[-1]
            urllib.request.urlretrieve(url, savePath + name)
print("Images downloaded")