import numpy as np
import cv2
cap=cv2.VideoCapture(0)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
while(True):
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([150,0,192]),np.array([159,127,255]))+cv2.inRange(hsv,np.array([0,0,255]),np.array([0,0,255]))+cv2.inRange(hsv,np.array([90,0,255]),np.array([90,10,255]))
    mask=cv2.erode(mask,skinkernel,iterations=1)
    mask=cv2.dilate(mask,skinkernel,iterations=1)
    mask=cv2.GaussianBlur(mask,(15,15),1)
    circles=cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=1,maxRadius=50)
    if isinstance(circles,np.ndarray):
        circles=np.uint16(np.around(circles))
        cv2.imwrite('frames/frame_'+str(cap.get(0))+'.png',frame)
        for i in circles[0,:]:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,0,255),-1)
    cv2.imshow('frame',frame)
    cv2.imshow('hsv',hsv)
    cv2.imshow('mask',mask)
    if cv2.waitKey(1) & 0xFF==27:
        cv2.imwrite('frames/frame_'+str(cap.get(0))+'.png',frame)
        break
cap.release()
cv2.destroyAllWindows()