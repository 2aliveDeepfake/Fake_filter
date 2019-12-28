import cv2
import numpy as np
import sys, os

#학습할 이미지 변수가 train입니다.
#work변수가 필터를 통과한 후 이미지입니다.

#이미지 불러오기
folder_path = "images/"
folder_list = os.listdir(folder_path)


for item in folder_list:  # 각 폴더 이름(hood_T)의 파일이름 얻기

    train = cv2.imread(folder_path+item)
    print(folder_path+item)
    #Emboss
    tmp = train.copy()
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)

    #Histogram_Equalization
    cv2.equalizeHist(tmp,tmp)
    tmp = cv2.cvtColor(tmp,cv2.COLOR_GRAY2RGB)

    #Bilateral
    tmp = cv2.bilateralFilter(tmp,30,102,102)

    #Gamma_correction
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i/255.0, 1.0/(30/10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    #Brightness&Contrast
    var_Brightness = 70 - 100
    var_Contrast = 200 - 100

    if(var_Contrast > 0):
        delta = 127.0 * var_Contrast / 100
        a = 255.0 / (255.0 - delta * 2)
        b = a * (var_Brightness - delta)
    else:
        delta = -128.0 * var_Contrast / 100
        a = (256.0 - delta * 2) / 255.0
        b = a * var_Brightness + delta

    tmp = tmp*a + b

    #필터 통과한 이미지 변수에 넣기
    work = tmp.copy()
    cv2.imwrite("output/"+item, work)
    #이미지 출력
    # cv2.imshow("aaa", work)
    # cv2.waitKey()