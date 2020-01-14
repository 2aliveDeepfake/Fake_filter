import cv2
import numpy as np
import sys, os

#학습할 이미지 변수가 train입니다.
#work변수가 필터를 통과한 후 이미지입니다.

# 폴더 내 이미지 불러오기
folder_path = "fake_low_tone/"
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

    # Saturation_add
    img_width = tmp.shape[0]
    img_height = tmp.shape[1]
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2HSV)
    for i in range(img_width):
        for j in range(img_height):
            tmp[i][j][1] += 30
    tmp = cv2.cvtColor(tmp, cv2.COLOR_HSV2RGB)

    #필터 적용 완료 =================================

    #필터 통과한 이미지 변수에 넣기
    work = tmp.copy()
    output_dir="tone_output/"
    output_path = output_dir+item

    # output 경로가 없으면 폴더를 생성합니다.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cv2.imwrite(output_path, work)

    #이미지 출력
    # cv2.imshow("aaa", work)
    # cv2.waitKey()