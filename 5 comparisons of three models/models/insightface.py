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

import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
assert insightface.__version__>='0.6'
import collections
from multiprocessing import Pool
import os
import time
import datetime

parser = argparse.ArgumentParser(description='insightface app test')
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=128, type=int, help='detection size')
args = parser.parse_args()
app = FaceAnalysis(name="buffalo_l",providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))  

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
    rstPath = "/home/rebecca/face/FACE/new/pair_results_insight2/" + str(name)+".txt"

    end =amount
    if start+per <= amount:
        end = start+per
    
    for i in range(start, end):
        pairs = img_pairs[i]
        pairs_result = open(rstPath, "a+")
        pic_1 = pairs[0].replace(".jpg","")
        pic_2 = pairs[1].replace(".jpg","")
    
        img_1 = ins_get_image(pic_1)
        img_2 = ins_get_image(pic_2)
        faces_1 = app.get(img_1)
        faces_2 = app.get(img_2)

        for face in faces_1:
            vector1 = face.embedding
            vector1 = vector1.reshape(1,-1)
        for face in faces_2:
            vector2 = face.embedding
            vector2 = vector2.reshape(1,-1)

        sim = np.abs(cosine_similarity(vector1,vector2))

        pic1 = pic_1.split("/")[-2] + "-"+ pic_1.split("/")[-1]
        pic2 = pic_2.split("/")[-2] + "-"+ pic_2.split("/")[-1]
        pairs_result.writelines(pic1+", "+ pic2 + ", " + "%.4f" % sim+ "\n")
        pairs_result.close()
   
tic = time.perf_counter()
with Pool() as p:
    p.map(to_parallel, workList)
#to_parallel(0)
toc = time.perf_counter()
timePath = "/home/rebecca/face/FACE/new/pair_results_insight2/time.txt"
time_result = open(timePath, "a+")
time_result.writelines(str(datetime.datetime.now())+ "   ")
time_result.writelines(f"Time: {toc - tic:0.4f} seconds")
time_result.writelines("\n")
time_result.close()
