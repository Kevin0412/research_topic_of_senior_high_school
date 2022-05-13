import cv2
import numpy as np

def led_in_window(wcx,wcy,wsx,wsy,ledx,ledy):#led灯是否在窗户识别框中
    aa=wcx*1280
    bb=wcy*720
    cc=wsx*1280
    dd=wsy*720
    left=aa-cc/2
    right=aa+cc/2
    up=bb-dd/2
    down=bb+dd/2
    if (ledx>=left and ledx<=right) and (ledy>=up and ledy<=down):
        return [False,int(left),int(up),int(right),int(down)]
    else:
        return [True,int(left),int(up),int(right),int(down)]

vc = cv2.VideoCapture('VID_20220120_200246.mp4')#原视频
video = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc('I', '4', '2', '0'),30,(1280,720))
n = 1
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
timeF = 1
cxy=[]
while rval: 
    rval, frame = vc.read()
    if (n % timeF == 0) and rval:
        with open('labels\\VID_20220120_200246_'+str(n)+'.txt',"r",encoding='utf-8') as F:#yolo_v5输出的识别框的.txt文件
            d2=F.read()
            d1=d2.split('\n')
            data=[]
            for d in d1:
                data.append(d.split(' '))
        unclear=cv2.blur(frame,(40,40))
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,np.array([112,30,192]),np.array([155,100,255]))#提取目标颜色
        mask=cv2.erode(mask,skinkernel,iterations=1)
        mask=cv2.dilate(mask,skinkernel,iterations=1)
        conts,hrc=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        bigconts=[]
        for cont in conts:
            M=cv2.moments(cont)
            cx=M['m10']/M['m00']
            cy=M['m01']/M['m00']
            for d in data:
                if d[0]=='16':
                    LIW=led_in_window(float(d[1]),float(d[2]),float(d[3]),float(d[4]),cx,cy)
                    if not LIW[0]:
                        bigconts.append(cont)
        c=[]
        for bigcnt in bigconts:
            M=cv2.moments(bigcnt)
            cx=M['m10']/M['m00']
            cy=M['m01']/M['m00']
            c.append((cx,cy))
        cxy.append(c)
        for c1 in cxy[-1]:
            if n!=1:
                for c2 in cxy[-2]:
                    if (c1[0]-c2[0])**2+(c1[1]-c2[1])**2<208:#前后两点间距离较近
                        for d in data:
                            if d[0]=='16':
                                LIW=led_in_window(float(d[1]),float(d[2]),float(d[3]),float(d[4]),c1[0],c1[1])
                                if LIW[0] and (float(d[3])*1280>175 or float(d[4])*720>175):
                                    frame[LIW[2]:LIW[4],LIW[1]:LIW[3]]=unclear[LIW[2]:LIW[4],LIW[1]:LIW[3]]
        cv2.imshow('frame',frame)
        video.write(frame)
        cv2.waitKey(1)
    n = n + 1
video.release()
vc.release()
