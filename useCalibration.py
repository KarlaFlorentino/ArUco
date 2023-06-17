import numpy as np
import cv2 as cv
import glob

cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

############## UNDISTORTION #####################################################


print('Camera matrix\n', cameraMatrix)
print('Dist\n', dist)


images = glob.glob('img/*.jpg')
for image in images:
    img = cv.imread(image)
    
    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    cv.imwrite(image.replace(".jpg", "_undistortion.jpg"), dst)