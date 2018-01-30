import cv2
import numpy as np

face = cv2.imread('./face.png')
eye = cv2.imread('./eye.png',-1)


head_width=35/2

while True:
    left_eye_rate = 0.3
    right_eye_rate = 0.7

    x,y = 0.,0.

    x = abs(max(left_eye_rate,right_eye_rate)-0.5)*head_width
    x *= 1 if left_eye_rate > right_eye_rate else -1

    rate_mim = 1.0
    cc = np.float32([[rate_mim,0,int(0+x)],[0,1,int(0+y)]])
    eye_moved = eye.copy()
    eye_moved = cv2.warpAffine(eye,cc,(eye.shape[1],eye.shape[0]))

    eye_na = cv2.cvtColor(eye_moved,cv2.COLOR_BGRA2BGR)

    mask = eye_moved[:,:,3]
    mask = mask.reshape((mask.shape[0],mask.shape[1],1))
    mask = np.uint8(np.tile(mask,(1,1,3)) / 225.)

    ds = eye_na*mask
    face_end = face.copy()
    face_end += ds


    cv2.imshow('face',face_end)



    key = cv2.waitKey(5)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
