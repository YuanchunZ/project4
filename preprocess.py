import pandas as pd
import shutil
import os

df = pd.read_csv("./data/structure.csv")
classes = set(df['Template Name'])
cla2num = {}
for i,k in enumerate(classes):
    cla2num[k] = i

label = df['Template Name']
paths = df['Image']

for p,l in zip(paths, label):
    targetPath = "./data/training/"+str(cla2num[l])
    if(not os.path.exists(targetPath)):
        os.makedirs(targetPath)
    shutil.copy(p, targetPath)
    


