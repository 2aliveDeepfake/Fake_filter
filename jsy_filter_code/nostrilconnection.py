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

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    #THRESH_TOZERO
    ret, tmp = cv2.threshold(tmp, 45,0, cv2.THRESH_TOZERO)

    # Gamma_correction
    gamma = 50
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)


    # 필터 통과한 이미지 변수에 넣기
    work = tmp.copy()


    #결과 이미지를 보여준다.
    cv2.imshow('aaa', work)
    cv2.waitKey()


    #필터를 통과한 이미지를 해당 경로의 폴더에 넣어준다.
    #원본이미지명 앞에 R_을 추가해서 저장한다.
    #print(imgfile_name[-1])
    # path = "C:\\Users\\jsych\\Desktop\\Results\\R_" + imgfile_name[-1]
    # cv2.imwrite(path,work)