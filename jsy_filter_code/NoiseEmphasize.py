import cv2
import numpy as np



train = cv2.imread("C:/Users/jsych/Desktop/Images/facebook_4323.jpg")
tmp = train.copy()
tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
result = cv2.equalizeHist(tmp)
result = cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)

tmp2 = result.copy()
tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_RGB2GRAY)
Kernel = np.array([[0,-5,0],[-5,21,-5],[0,-5,0]])
grad = cv2.filter2D(tmp2,cv2.CV_8U,Kernel)
work = grad.copy()

cv2.imshow("aaa",work)
cv2.waitKey()