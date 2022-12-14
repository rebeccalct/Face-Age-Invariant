import argparse
import cv2
import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os 
import shutil

assert insightface.__version__>='0.3'

# general
parser = argparse.ArgumentParser(description='insightface app test')
parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=128, type=int, help='detection size')
args = parser.parse_args()

model_pack_name = 'buffalo_l'
app = FaceAnalysis(name=model_pack_name)
app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))

# Move pictures to another file and then check them manually to ensure the accuracy
destination = 'F:\\MS\\Project\\face_search_env\\Test_gender\\wrong_gender'
path = 'F:\\MS\\Project\\face_search_env\\Test_gender\\AgeDB_copy'

file = os.listdir(path)

for i in file:
    name = i.split('.')[0] 
    path1 = path + '\\' + name
    img = ins_get_image(path1)
    faces = app.get(img)
    name1 = name.split('_')[3] 
    name1 = name1.capitalize()
    print(path1)
    for k in range(len(faces)):
        face = faces[k]   
        if face.sex != name1:
            source = path1 + '.jpg'
            shutil.move(source, destination)
