import cv2
import os
import json

dir = 'dataset/'

datalists = os.listdir(dir)
json_datas = open(dir+'dataset_back.json','r')
json_datas = json.load(json_datas)

for i in json_datas:
    pict = cv2.imread(dir+i)
    cv2.putText(pict,i,(100,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
    for j in json_datas[i]:
        cv2.circle(pict, (json_datas[i][j]['y'],json_datas[i][j]['x']), 10, (0, 0, 255), -1)
    while True:
        key = cv2.waitKey(1)
        if key == ord('n'):
            break
        if key == ord('q'):
            exit(0)
        cv2.imshow('pict',pict)
