# Program to create custom ArUco dictionary using OpenCV and detect markers using webcam
# original code from: http://www.philipzucker.com/aruco-in-opencv/
# Modified by Iyad Aldaqre
# 12.07.2019

import numpy as np
import cv2
import cv2.aruco as aruco
import glob #usada para manipular os arquivos
import xml.etree.cElementTree as ET
import os

cameraMatrix = np.load("Original camera matrix.npy")
dist = np.load("Distortion coefficients.npy")
newCameraMatrix = np.load("Optimal camera matrix.npy")

markerLength = 50

auxParam = 0 #3 #conjunto de parametros 

arucoParams = aruco.DetectorParameters_create()

if(auxParam != 0):
    arucoParams.minMarkerPerimeterRate = 0.09 #default 0.03
    arucoParams.polygonalApproxAccuracyRate = 0.1 #default 0.03 0.1
    arucoParams.maxErroneousBitsInBorderRate = 0.30 #default 0.35

if (auxParam == 1):
    arucoParams.maxMarkerPerimeterRate = 2.0 #default 4.0
    arucoParams.adaptiveThreshWinSizeMax = 393  #default 23 
    arucoParams.adaptiveThreshWinSizeMin = 83  #default 3 83

elif (auxParam == 2):
    arucoParams.maxMarkerPerimeterRate = 2.0 #default 4.0
    arucoParams.adaptiveThreshWinSizeMin = 5
    arucoParams.cornerRefinementMethod = 2
    
elif (auxParam == 3):
    arucoParams.maxMarkerPerimeterRate = 1.85 #default 4.0
    arucoParams.adaptiveThreshWinSizeMax = 755 
    arucoParams.adaptiveThreshWinSizeMin = 55

    #arucoParams.adaptiveThreshWinSizeMax = 95  
    arucoParams.adaptiveThreshConstant = 0.5
    #arucoParams.cornerRefinementMethod = 2


# define an empty custom dictionary with 
aruco_dict = aruco.custom_dictionary(0, 3, 1)
# add empty bytesList array to fill with 3 markers later
aruco_dict.bytesList = np.empty(shape = (1, 2, 4), dtype = np.uint8)

# add new marker(s)
mybits = np.array([[1,0,1],[1,0,0],[1,1,1]], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)

# save marker images
#for i in range(len(aruco_dict.bytesList)):
#    cv2.imwrite("custom_aruco_" + str(i) + ".png", aruco.drawMarker(aruco_dict, i, 128))


#https://stackoverflow.com/questions/46363618/aruco-markers-with-opencv-get-the-3d-corner-coordinates
def getCornersInCameraWorld(markerLength, rvec, tvec):
    ##Pinhole camera model
    half_side = markerLength / 2
    rot_mat, _ = cv2.Rodrigues(rvec)

    #print("Rot_mat: \n", rot_mat, "\n")

    rot_mat_t = np.transpose(rot_mat)

    #print("Rot_mat_t: \n", rot_mat_t, "\n")

    # E-0
    tmp = rot_mat_t[:, 0]
    camWorldE = np.array([tmp[0] * half_side,
                          tmp[1] * half_side,
                          tmp[2] * half_side])

    # F-0
    tmp = rot_mat_t[:, 1]
    camWorldF = np.array([tmp[0] * half_side,
                          tmp[1] * half_side,
                          tmp[2] * half_side])

    tvec_3f = np.array([tvec[0][0][0],
                        tvec[0][0][1],
                        tvec[0][0][2]])

    nCamWorldE = np.multiply(camWorldE, -1)
    nCamWorldF = np.multiply(camWorldF, -1)

    ret = np.array([tvec_3f, tvec_3f, tvec_3f, tvec_3f])
    ret[0] += np.add(nCamWorldE, camWorldF)
    ret[1] += np.add(camWorldE, camWorldF)
    ret[2] += np.add(camWorldE, nCamWorldF)
    ret[3] += np.add(nCamWorldE, nCamWorldF)

    return ret



# open video capture from (first) webcam
#cap = cv2.VideoCapture(0)

VideoCap = False

path = 'D:/Downloads/Feijao/'

for i in range(700, 701):
    #for i in range(126, 127):
    leafNumber = str(i).zfill(3)
    print(leafNumber)

    # Obtém o nome de todas as imagens na pasta
    images = glob.glob(path + leafNumber + '/*.jpg')

    for fname in images:
        print(fname)
        frame = cv2.imread(fname)
        #lists of ids and the corners beloning to each id
        if(auxParam == 0): corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict)
        else: corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)

        if np.all(ids is not None):  # If there are markers found by detector

            #***************************************************************
            #Analize detection(mask) quality

            # draw markers on farme
            aruco.drawDetectedMarkers(frame, corners, ids)

            corners_copy = corners[0][0]
            cv2.circle(frame, tuple(np.int32([corners_copy[0][0], corners_copy[0][1]])), radius=20, color=(0, 0, 255), thickness=-1)
            cv2.circle(frame, tuple(np.int32([corners_copy[1][0], corners_copy[1][1]])), radius=20, color=(0, 255, 0), thickness=-1)
            cv2.circle(frame, tuple(np.int32([corners_copy[2][0], corners_copy[2][1]])), radius=20, color=(255, 0, 0), thickness=-1)
            cv2.circle(frame, tuple(np.int32([corners_copy[3][0], corners_copy[3][1]])), radius=20, color=(0, 255, 255), thickness=-1)
            
            print('Corners:\n', corners)

            # Estimate the pose of the object using estimatePoseSingleMarkers
            rvec_sm, tvec_sm, markerPoints = aruco.estimatePoseSingleMarkers(
                corners,
                markerLength,
                cameraMatrix,
                dist
            )
            (rvec_sm - tvec_sm).any()  # get rid of that nasty numpy value array error
            rotation_mat_sm, _ = cv2.Rodrigues(rvec_sm)

            print('TVEC:\n', tvec_sm)
            print('RVEC:\n', rvec_sm)

            aruco.drawAxis(frame, cameraMatrix, dist, rvec_sm, tvec_sm, markerLength)  # Draw axis

            #RET with SingleMarkers GREEN
            ret = getCornersInCameraWorld(markerLength, rvec_sm, tvec_sm)
            print('RET with SingleMarkers:\n', ret)
            '''for p in ret:
                aux = np.dot(cameraMatrix, p) / p[2]
                aux_woZ = np.delete(aux, -1, axis=0)
                cv2.circle(frame, tuple(np.int32(aux_woZ)), radius=20, color=(255, 255, 255), thickness=-1)
            '''

            # Estimate the pose of the object using solvePNP
            # Calculate the 3D coordinates of the Aruco marker
            object_points = np.array([[-markerLength/2,  markerLength/2, 0], 
                                    [ markerLength/2,  markerLength/2, 0], 
                                    [ markerLength/2, -markerLength/2, 0], 
                                    [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)
            
            #image_points = corners[ids == marker_id][0]
            success, rvec_solve, tvec_solve = cv2.solvePnP(object_points, 
                                            corners[0], 
                                            cameraMatrix, 
                                            dist, 
                                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
            print('Sucess: ', success)
            rotation_mat_solve, _ = cv2.Rodrigues(rvec_solve)

            print('TVEC:\n', tvec_solve)
            print('RVEC:\n', rvec_solve)


            #RET with solvePnP BLACK
            tvec_solve = tvec_solve.flatten()
            rvec_solve = rvec_solve.flatten()
            ret = getCornersInCameraWorld(markerLength, rvec_solve, [[tvec_solve]])
            print('RET with solvePnP:\n', ret)
            '''for p in ret:
                aux = np.dot(cameraMatrix, p) / p[2]
                aux_woZ = np.delete(aux, -1, axis=0)
                cv2.circle(frame, tuple(np.int32(aux_woZ)), radius=20, color=(0, 0, 0), thickness=-1)
            '''

            mask = mask = np.zeros(frame.shape, np.uint8)
            copyCorners = np.array(corners[0][0], dtype=np.int32)
            copyCorners = copyCorners.reshape(-1,1,2)

            color = (255, 255, 255)
            mask = cv2.fillPoly(mask, [copyCorners], color)
        
            imgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0) #testar otsu exibir
            #_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), 10) 

            #create dir "marker"
            if not os.path.exists(str(fname.split("\\")[0]) + "\marker"):
                os.makedirs(str(fname.split("\\")[0]) + "\marker")

            newPath = fname.replace(leafNumber + "\\", leafNumber + "\marker\\")
            cv2.imwrite(newPath, frame)

            #***************************************************************
            #Normalize corner points
            
            h = 4624
            normalize = copyCorners/h

            #for element in normalize:
            #    print(element[0][0], element[0][1])


            headerXML = '''<annotation>\n  <filename>''' + str(fname.split("\\")[1]) + '''</filename>\n  <object>\n    <leaf> ''' + leafNumber + ''' </leaf>\n    <corners>\n'''

            i = 1
            elements = ""
            for element in normalize:
                elements = elements+ '      <x%s>%s</x%s>\n        <y%s>%s</y%s>\n'% (i, element[0][0], i , i , element[0][1], i)
                i += 1
            footerXML = '''    </corners>\n  </object>\n</annotation>'''

            XML = (headerXML+elements+footerXML)
            #print(XML)

            #outFile = open(newPath.replace("jpg","xml"),"w")
            #outFile.write(XML)
            #outFile.close()
        else:
            print("Error")
        # resize frame to show even on smaller screens
        #mask = cv2.resize(mask, (0,0), fx = 0.2, fy = 0.2)
        #frame = cv2.resize(frame, (0,0), fx = 0.2, fy = 0.2)
                       
    # Displaying the image
    '''while(1):
        cv2.imshow('Mask', mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''

# When everything done, release the capture
#cap.release()
cv2.destroyAllWindows()