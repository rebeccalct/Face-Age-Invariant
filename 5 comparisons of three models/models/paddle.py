# read pair of images
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

import insightface_paddle as face
import paddle
import paddle.nn as nn
import logging
import cv2
from multiprocessing import Pool
import os
import time

logging.basicConfig(level=logging.INFO)
parser = face.parser()
args = parser.parse_args()
args.det = True
args.rec = True
args.index = "/home/rebecca/face/FACE/index.bin"
predictor = face.InsightFace(args) # Here we can change the the loss functions

amount = len(img_pairs)
per = int(amount/40)
workList = []
record = 0
while(record<amount):
    workList.append(record)
    record += per  

#res = predictor.predict(input_path, print_info=False)
def to_parallel(start):
    name = os.getpid()
    rstPath = "/home/rebecca/face/FACE/new/pair_result_paddle2/" + str(name)+".txt"
    #for pairs in img_pairs::
    #end = amount #######################################################
    end =amount
    if start+per <= amount:
        end = start+per############################################
    #end = start +3
    
    for i in range(start, end):
        pairs = img_pairs[i]
        pairs_result = open(rstPath, "a+")
        pic_1 = pairs[0]
        pic_2 = pairs[1]
        sim = predictor.build_embedding(pic_1,pic_2)
        pic1 = pic_1.split("/")[-2] + "-"+ pic_1.split("/")[-1].replace(".jpg","")
        pic2 = pic_2.split("/")[-2] + "-"+ pic_2.split("/")[-1].replace(".jpg","")
        pairs_result.writelines(pic1+", "+ pic2 + ", " + "%.4f" % sim+ "\n")
        pairs_result.close()        

tic = time.perf_counter()
with Pool() as p:
    p.map(to_parallel, workList)
#to_parallel(0)
toc = time.perf_counter()
timePath = "/home/rebecca/face/FACE/new/pair_result_paddle2/time.txt"
time_result = open(timePath, "a+")
time_result.writelines(str(datetime.datetime.now())+ "   ")
time_result.writelines(f"Time: {toc - tic:0.4f} seconds")
time_result.writelines("\n")
time_result.close()
