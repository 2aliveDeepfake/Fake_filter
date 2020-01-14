import cv2
import glob
import numpy as np

#작업할 이미지 경로
img_dir = "C:\\Users\\jsych\\Desktop\\Images\\*"
file_list = glob.glob(img_dir)
#.jpg파일만 리스트에 넣는다.
img_list = [file for file in file_list if file.endswith(".jpg")]
#작업할 이미지의 개수
cnt_img = len(img_list) - 1

for i in range(cnt_img):
    imgfile_name = img_list[i].split("\\")
    train = cv2.imread(img_list[i])
    tmp = train.copy()

    #CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(11, 11))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)


    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 7, 51, 51)

    # Gamma_correction
    gamma = 80
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    #THRESH_TOZERO
    ret, tmp = cv2.threshold(tmp, 220,0, cv2.THRESH_TOZERO)


    # Brightness&Contrast
    var_Brightness = 0 - 100
    var_Contrast = 200 - 100

    if (var_Contrast > 0):
        delta = 127.0 * var_Contrast / 100
        a = 255.0 / (255.0 - delta * 2)
        b = a * (var_Brightness - delta)

    else:
        delta = -128.0 * var_Contrast / 100
        a = (256.0 - delta * 2) / 255.0
        b = a * var_Brightness + delta

    tmp = tmp * a + b


    # 필터 통과한 이미지 변수에 넣기
    work = tmp.copy()

    cv2.imshow('bbb', train)
    cv2.imshow('aaa', work)
    cv2.waitKey()

    #print(imgfile_name[-1])
    # path = "C:\\Users\\jsych\\Desktop\\Results\\R_" + imgfile_name[-1]
    # cv2.imwrite(path,work)
