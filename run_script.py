import os
import subprocess

# File that used to run the script code

documents = []
path = "./images/"
for document in os.listdir(path):
    documents.append(path+str(document))

print(documents)
documents.sort()

for i in range(len(documents)):
    if i >= len(documents) -1:
        break
    num = str(i)
    if i < 10:
        num = "0" + str(i)

    instruction =  "./deepmatching_after_match/deepmatching " + str(documents[i]) + " " + str(documents[i+1])+ " | ./DeepFlow2_after_match/deepflow2 " + str(documents[i]) + " " + str(documents[i+1]) + " ./flow/"+ num +".sintel.flo -match -middlebury"
    completed = subprocess.run(instruction,shell=True)
    print('returncode:', completed.returncode)


