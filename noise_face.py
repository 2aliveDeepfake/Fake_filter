import cv2
import numpy as np
import sys, os

#학습할 이미지 변수가 train입니다.
#work변수가 필터를 통과한 후 이미지입니다.

# 폴더 내 이미지 불러오기
folder_path = "noise_real/"
folder_list = os.listdir(folder_path)
output_dir="noise_real_output/"

for item in folder_list:  # 폴더의 파일이름 얻기

    train = cv2.imread(folder_path+item)
    print(folder_path+item)
    tmp = train.copy()
    # 필터 적용 부분 ===================================
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    result = cv2.equalizeHist(tmp)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    tmp2 = result.copy()
    tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_RGB2GRAY)
    Kernel = np.array([[0, -5, 0], [-5, 21, -5], [0, -5, 0]])
    grad = cv2.filter2D(tmp2, cv2.CV_8U, Kernel)
    work = grad.copy()

    #필터 적용 완료 =================================

    #필터 통과한 이미지 변수에 넣기
    #work = tmp.copy()
    output_path = output_dir+item

    # output 경로가 없으면 폴더를 생성합니다.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cv2.imwrite(output_path, work)

    #이미지 출력
    # cv2.imshow("aaa", work)
    # cv2.waitKey()