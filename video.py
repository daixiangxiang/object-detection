#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from frcnn import FRCNN

frcnn = FRCNN()
#-------------------------------------#
#   调用摄像头
#capture=cv2.VideoCapture('E:/python/faster-rcnn-pytorch-master/video/passageway1-c1.avi')
capture=cv2.VideoCapture('E:/FFOutput/try.mp4')
#-------------------------------------#
#capture=cv2.VideoCapture(0)
t0= time.time()
fps = 0.0
L=list()
L1=list()
top0=0
left0=0
bottom0=0
right0=0

while capture.isOpened():
    # 读取某一帧
    t1 = time.time()
    ref,frame=capture.read()
    if not ref:
        capture.release()
        cv2.destroyAllWindows()
        break

    # 格式转变，BGRtoRGB
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame,t2, top, left, bottom, right= frcnn.detect_image(frame)

    part1=top-top0
    part2=left-left0
    part3 = bottom -bottom0
    part4 = right - right0
    ave=(part1+part2+part3+part4)/4

    t2=np.array(t2)
    frame=np.array(frame)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    L.append(fps)
    #print("fps= %.2f"%(fps))
    print('ave',ave)
    if (abs(ave) > 1):
        print('\033[1;31m-----------------------------------------------\033[0m')
        print('\033[1;31m 检测到猫所耗费的时间：',t2-t0,'秒\033[0m')
        print('\033[1;31m-----------------------------------------------\033[0m')
    # if 15<t2-t0 <35 :
    #     L1.append(t2-t0)
    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("video",frame)
    c= cv2.waitKey(1) & 0xff 
    if c==27:
        capture.release()
        cv2.destroyAllWindows()
        break
    top0=top
    left0=left
    bottom0=bottom
    right0=right

# print('平均检测时间：',sum(L1)/len(L1),'秒')
# print('平均处理帧率', sum(L) / len(L))