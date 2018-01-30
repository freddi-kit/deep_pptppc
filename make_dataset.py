import cv2
import os
import json

# マウスイベント時に処理を行う
class mouseInfo:
    def __init__(self,winName):
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(winName,self.callBack)
        self.ix = None
        self.iy = None

    def callBack(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.ix,self.iy  = x,y

    def getXY(self):
        return self.ix,self.iy

    def init_next(self):
        self.ix,self.iy  = None,None

dir = 'dataset/'

datalists = os.listdir(dir)
json_datas = open(dir+'dataset.json','w')

data_c = dict()

keys = 5

key_names = ['left_eye','right_eye','norse','left_s','right_s']

mf = mouseInfo('pict')

data_num = 0

for i in datalists:
    if i[-4:] == '.png':
        pict = cv2.imread(dir+i)
        set_part = 0
        data_c[i] = dict()
        while True:
            c = pict.copy()
            cv2.putText(c,key_names[set_part],(100,100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)
            cv2.putText(c,str(data_num+1)+'/'+str(len(datalists)),(100,c.shape[0]-100),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),2)

            sety,setx = mf.getXY()
            if not setx == None:
                cv2.circle(c, (sety,setx), 10, (0, 0, 255), -1)

            key = cv2.waitKey(1)
            if key == ord('z'):
                data_c[i][set_part] = dict()
                if setx == None:
                    data_c[i][set_part]['x'] = -1
                    data_c[i][set_part]['y'] = -1

                else:
                    data_c[i][set_part]['x'] = setx
                    data_c[i][set_part]['y'] = sety

                set_part += 1
                mf.init_next()
                if set_part == keys:
                    set_part=0
                    data_num+=1
                    break

            elif key == ord('x'):
                if set_part > 0:
                    set_part -= 1
                mf.init_next()

            elif key == ord('c'):
                json.dump(data_c,json_datas,indent=4, separators=(',', ': '))
                exit(0)
                break
            cv2.imshow('pict',c)

json.dump(data_c,json_datas,indent=4, separators=(',', ': '))
