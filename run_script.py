import os
import subprocess

#for i in range(2):
#    os.system("python test.py")

documents = []
path = "./images/"
for document in os.listdir(path):
    documents.append(path+str(document))

print(documents)

for i in range(len(documents)):
    if i >= len(documents) -1:
        break
    instruction =  "/deepmatching " + str(documents[i]) + " " + str(documents[i+1])+ " | ./deepflow2 " + str(documents[i]) + " " + str(documents[i+1]) + " ./flow/"+ str(i)+".sintel.flo -match -middlebury"
    completed = subprocess.run(instruction,shell=True)
    print('returncode:', completed.returncode)


