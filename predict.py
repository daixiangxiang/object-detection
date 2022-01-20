'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
import numpy as np
from yolo import YOLO

yolo = YOLO()
#t0= time.time()
# img='E:/FFOutput/640x480_88k.jpg'
#
# image = Image.open(img)
# image = image.convert("RGB")
# r_image,t2 = yolo.detect_image(image)
# t2 = np.array(t2)
# print('\033[1;31m-----------------------------------------------\033[0m')
# print('\033[1;31m 检测到自行车所耗费的时间：', t2, '秒\033[0m')
# print('\033[1;31m-----------------------------------------------\033[0m')
#
# r_image.show()

while True:
    #img = input('Input image filename:')
    img = r"E:\data\VIRA\image2\image_26.jpg"
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
