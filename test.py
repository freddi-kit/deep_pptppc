import chainer
from chainer import function as F
from chainer import Variable,cuda,optimizers,serializers
from make_gaussian import make_dataset
from cpm import CPM
import numpy as np
from PIL import Image
import os,sys
import cv2

GPU = 0

net = CPM(n_point=5,n_stage=3)

xp = np
if GPU>=0:
    cuda.get_device(GPU).use()
    xp = cuda.cupy
    net.to_gpu(GPU)

dir = sys.argv[1]
pict_list = os.listdir(dir+'/')
serializers.load_npz(sys.argv[2], net)


face = cv2.imread('./face.png')
eye = cv2.imread('./eye.png',-1)


head_width=37/2


def max_coo(heatmap):
    max_pixel = 0
    coo = (-1,-1)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if max_pixel < heatmap[i][j]:
                max_pixel = heatmap[i][j]
                coo = (i,j)
    return coo

for i in pict_list:
    if i[-4:] == '.png':
        img = Image.open(dir+'/'+i)
        img = np.asarray(img,dtype=np.float32)/255.
        img = img.transpose(2,0,1)
        x_train = Variable(xp.asarray([img],dtype=np.float32))

        y = net.predict(x_train)
        
        y_cpu = cuda.to_cpu(y.data)[0]
        y_cpu = np.clip(y_cpu,0.0,1.0)

        l_eye = max_coo(y_cpu[3])
        r_eye = max_coo(y_cpu[1])
        nose  = max_coo(y_cpu[4])

        all_len = l_eye[1]-r_eye[1]

        left_eye_rate = (nose[1] - r_eye[1])/ all_len
        right_eye_rate = 1 - left_eye_rate
        
       

        face_go = cv2.imread(dir+'/'+i)
        #cv2.circle(pict, (json_datas[i][j]['y'],json_datas[i][j]['x']), 10, (0, 0, 255), -1)

  
        x,y = 0.,0.

        x = abs(max(left_eye_rate,right_eye_rate)-0.5)*head_width
        x *= 1 if left_eye_rate > right_eye_rate else -1

        rate_mim = 1.0
        cc = np.float32([[rate_mim,0,int(0+x)],[0,1,int(0+y)]])
        eye_moved = eye.copy()
        eye_moved = cv2.warpAffine(eye_moved,cc,(eye.shape[1],eye.shape[0]))

        mask = eye_moved[:,:,3]
        mask = mask.reshape((mask.shape[0],mask.shape[1],1))
        mask = np.int8(np.tile(mask,(1,1,3)) / 225.)
        mask_invert = -mask+1
        face_end = mask+mask_invert*face
        
        cv2.line(face_end, (round(face_end.shape[1]/2),0), (round(face_end.shape[1]/2),round(face_end.shape[0])), (255, 0, 0), 2)
 


        face_go[:face_end.shape[0],:face_end.shape[1],:]=face_end
        cv2.circle(face_go, (l_eye[1]*8,l_eye[0]*8), 3, (0, 0, 255), -1)
        cv2.circle(face_go, (r_eye[1]*8,r_eye[0]*8), 3, (0, 0, 255), -1)
        cv2.circle(face_go, (nose[1]*8,nose[0]*8), 3, (0, 255, 0), -1)
        
        
        cv2.imshow('face_r',face_go)

        key = cv2.waitKey(5)
        cv2.imwrite('res/res_'+i,face_go)
        if key == ord('q'):
            #cv2.destroyAllWindows()
            break


        '''y_image = np.clip(y_image,0.0,1.0).transpose(1,2,0)*255.
        y_image = np.tile(y_image,(1,1,3))
        y_image = Image.fromarray(np.uint8(y_image))
        y_image = y_image.resize((y.data.shape[3]*8,y.data.shape[2]*8))
        y_image.save(dir+'/'+str(i)+'_a.png')'''

