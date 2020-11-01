#======================================================================
#- Created By: Rushil Verma
#- Project   : ImageMatcher
#- Version   : 1.0
#- Created on: 11/10/2020
#- Modified on: --/--/----
#- Details:
#- PURPOSE to Understand SIFT through video subject matching 
#-- Present code require video device to be connected to computer eg-WebCam
#-- Press C key on keyboard to capture Test Image to match with other images
#-- Good Matches will be represented through images graphs and its numeric count in console
#-- Press Q key on keyboard to Exit Program

##### IMPORTANT - RUN - "pip install --user opencv-contrib-python"
                         pip install opencv-python
                         pip install numpy
                         
#======================================================================

import cv2
#import matplotlib.pyplot as plt
import numpy as np

#to initialise video and SIFT alogoritms

vid = cv2.VideoCapture(0)
sift = cv2.xfeatures2d.SIFT_create()

print("Press C key to Capture test Image:")

#To Capture first testing image to match with others

while(True):
    ret0, img0=vid.read();
    cv2.imshow("Press C key to Capture Test Subject",img0)

    if cv2.waitKey(1) & 0xFF == ord('c'):   # Press C to Capture test subject
        img0=cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        break

#Detecting SIFT points in image
    
kp0, des0 = sift.detectAndCompute(img0,None)
img1x=cv2.drawKeypoints(img0,kp0,np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#mf.my_imshow(img1x)

while(True):
    
    #capture frames from video
    ret1, img1 =vid.read(); #img1 is frame
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)

    img2x=cv2.drawKeypoints(img1,kp1,np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #mf.my_imshow(img2x)

    # Brute Force Matcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0,des1, k=2)

    #----------Set threshold to apropriate value to identify no. of good matches needed to sucessfully Identification
    th=30

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    print('Good Matches : ',len(good),'               Press "Q" key exit')
    
    if(len(good)>=th):
        print("Object Matches with test image     -Threshold exceeded")
        
    img3 = cv2.drawMatchesKnn(img0,kp0,img1,kp1,good,np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # For Python API, flags are modified as -----
    # cv2.DRAW_MATCHES_FLAGS_DEFAULT
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
    # cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

    cv2.imshow('MATCHES - Press "Q" key exit',img3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):   # Press Q for Exit
        break

vid.release()
cv2.destroyAllWindows()
print("EXIT Triggered - 'Q' Key pressed")


print("------------------------------------------------------------")
print("-----Thanks for using Image Matcher 1.0 by Rushil Verma-----")
print("------------------------------------------------------------")