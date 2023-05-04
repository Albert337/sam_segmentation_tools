

import numpy as np
import os
import cv2

src=cv2.imread("images/8000.jpg")

data1=(np.loadtxt("images/result_mask/59.txt")/59.0).astype(np.int8)
data2=(np.loadtxt("images/result_mask/61.txt")/61.0).astype(np.int8)

data3=np.logical_and(data1,data2)

print(data1,data2)
