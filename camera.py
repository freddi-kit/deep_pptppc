import cv2

cap = cv2.VideoCapture(0)

i = 0
fc = 0
while True:
    ret,frame = cap.read()
    cv2.imshow('camera',frame)
    key = cv2.waitKey(5)
    
    if fc % 2 == 0:
        cv2.imwrite('dataset/'+str(i)+'.png',frame)
        i+=1
    fc+=1
    if key == ord('q'):
        break

cv2.destroyAllWindows()
