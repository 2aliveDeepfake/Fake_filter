import cv2
import glob
import numpy as np

#작업할 이미지 경로
img_dir = "real\\*"
file_list = glob.glob(img_dir)
#.jpg파일만 리스트에 넣는다.
img_list = [file for file in file_list if file.endswith(".jpg")]
#작업할 이미지의 개수
cnt_img = len(img_list) - 1
print(cnt_img)
for i in range(cnt_img):
    imgfile_name = img_list[i].split("\\")
    train = cv2.imread(img_list[i])
    tmp = train.copy()


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

# 필터 통과한 이미지 변수에 넣기
work = tmp.copy()


# 결과 이미지를 보여준다.
# cv2.imshow('aaa', work)
# cv2.waitKey()

# 필터를 통과한 이미지를 해당 경로의 폴더에 넣어준다.
# 원본이미지명 앞에 R_을 추가해서 저장한다.
path = "C:\\Users\\jsych\\Desktop\\Results\\R_" + imgfile_name[-1]
cv2.imwrite(path,work)
