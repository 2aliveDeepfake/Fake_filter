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

    

    cv2.imshow('bbb', train)
    cv2.imshow('aaa', work)
    cv2.waitKey()

    #print(imgfile_name[-1])
    # path = "C:\\Users\\jsych\\Desktop\\Results\\R_" + imgfile_name[-1]
    # cv2.imwrite(path,work)