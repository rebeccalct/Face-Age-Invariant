# create  the txt 
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import shutil
txt_file = open('C:\\Users\\rebeccalai\\OneDrive\\desktop\\FACE\\pairs.txt','w')
path = 'E:\\AgeDB-1-copy'
file = os.listdir(path)

for i in range(len(file)):
    path1 = path + '\\' + file[i]
    file1 = os.listdir(path1)
    l = []

    for j in range(len(file1)):
        path2 = path1 + '\\' + file1[j]
        for k in range(j+1, len(file1)):
            path3 = path1 + '\\' + file1[k]
            txt_file.writelines(path2 + ',' + path3 + '\n')

