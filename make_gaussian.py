from PIL import Image
import os
import json
import numpy as np

def gen_gaussian_heatmap(imshape, joint, sigma):
    x, y = joint
    grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
    grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
    grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
    gaussian_heatmap = np.exp(-grid_distance / sigma ** 2)
    return gaussian_heatmap




def make_dataset(path):
    dir = path

    datalists = os.listdir(dir)
    json_datas = open(dir+'dataset_back.json','r')
    json_datas = json.load(json_datas)
    x = []
    t = []

    list_dataset = []
    for i in json_datas:
        print(i)
        list_pict = []
        pict = np.asarray(Image.open(dir+i),dtype=np.float32)

        for j in json_datas[i]:
            gaussian = gen_gaussian_heatmap((int(pict.shape[0]/8),int(pict.shape[1]/8)),(int(json_datas[i][j]['y']/8),int(json_datas[i][j]['x']/8)),1)
            list_pict.append(gaussian)

        x.append(np.asarray(pict,dtype=np.float32).transpose((2,0,1))/255.)
        t.append(np.asarray(list_pict,dtype=np.float32))
    return x,t
