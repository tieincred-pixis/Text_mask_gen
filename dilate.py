import cv2  
import numpy as np  
import os

images = os.listdir('poc_res')


for image in images:
    print(image)
    if 'jpg' in image:
        img = cv2.imread(f'poc_res/{image}', 0)  
        kernel_ero = np.ones((3,3), np.uint8)  
        kernel_dil = np.ones((5,5), np.uint8)  
        img_erosion = cv2.erode(img, kernel_ero, iterations=1)  
        img_dilation = cv2.dilate(img, kernel_dil, iterations=1)  
        cv2.imwrite(f'poc_res/mask_out/{image}', img_dilation)  