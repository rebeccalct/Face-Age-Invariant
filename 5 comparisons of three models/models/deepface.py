# read pair of images
from sklearn.metrics.pairwise import cosine_similarity
pairs_txt_path = "/home/rebecca/face/FACE/new/pairs2.txt"

def get_img_pairs_list(pairs_txt_path):
    file = open(pairs_txt_path)
    img_pairs_list= []
    while 1:
        img_pairs = []
        line = file.readline().replace('\n','')
        if line == '':
            break
        line_list = line.split(',')
        img_pairs_list.append(line_list)
    return img_pairs_list

img_pairs =  get_img_pairs_list(pairs_txt_path)

import cv2
import pickle
from deepface import DeepFace
from multiprocessing import Pool
import os
import time
import numpy as np


amount = len(img_pairs)
per = int(amount/20)
workList = []
record = 0
while(record<amount):
    workList.append(record)
    record += per  

#res = predictor.predict(input_path, print_info=False)
def to_parallel(start):
    name = os.getpid()
    rstPath = "/home/rebecca/face/FACE/new/deepface2/" + str(name)+".txt"

    end =amount
    if start+per <= amount:
        end = start+per
    
    for i in range(start, end):
        pairs = img_pairs[i]
        pairs_result = open(rstPath, "a+")
        pic_1 = pairs[0]
        pic_2 = pairs[1]

        compare = DeepFace.verify(pic_1, pic_2, model_name = 'ArcFace', enforce_detection=False)
        distance = compare['distance']
        sim = np.abs(1-float(distance))

        pic1 = pic_1.split("/")[-2] + "-"+ pic_1.split("/")[-1].replace(".jpg","")
        pic2 = pic_2.split("/")[-2] + "-"+ pic_2.split("/")[-1].replace(".jpg","")
        pairs_result.writelines(pic1+", "+ pic2 + ", " + "%.4f" % sim+ "\n")
        pairs_result.close()   
   
tic = time.perf_counter()
with Pool() as p:
    p.map(to_parallel, workList)

#to_parallel(0)
toc = time.perf_counter()
timePath = "/home/rebecca/face/FACE/new/deepface2/time.txt"
time_result = open(timePath, "a+")
time_result.writelines(f"Time: {toc - tic:0.4f} seconds")
time_result.writelines("\n")
time_result.close()
