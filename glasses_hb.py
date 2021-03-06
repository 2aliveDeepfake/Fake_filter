import cv2
import numpy as np
import sys, os

#학습할 이미지 변수가 train입니다.
#work변수가 필터를 통과한 후 이미지입니다.

# 폴더 내 이미지 불러오기
folder_path = "no_glasses/"
folder_list = os.listdir(folder_path)

for item in folder_list:  # 폴더의 파일이름 얻기

    train = cv2.imread(folder_path+item)
    print(folder_path+item)
    tmp = train.copy()

    # 필터 적용 부분 ===================================
    # CLAHE
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

    # THRESH_TOZERO
    ret, tmp = cv2.threshold(tmp, 220, 0, cv2.THRESH_TOZERO)

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

    #필터 적용 완료 =================================

    #필터 통과한 이미지 변수에 넣기
    work = tmp.copy()
    output_dir="no_glasses_output/"
    output_path = output_dir+item

    # output 경로가 없으면 폴더를 생성합니다.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cv2.imwrite(output_path, work)

    #이미지 출력
    # cv2.imshow("aaa", work)
    # cv2.waitKey()