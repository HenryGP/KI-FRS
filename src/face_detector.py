"""
    Face detection module.
    Based on OpenCV sample.
    Code at:
    https://github.com/Itseez/opencv/blob/master/samples/python2/facedetect.py
"""

import freenect
import cv2 as cv
import frame_convert
import os.path
import numpy as np

#Global variables
keep_running = True
tilt_angle=None
r_color = (0,0,255)
picture_flag=False

#Window creation
cv.namedWindow('RGB')

#Path to .xml file
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +"/data/"
cascPath = data_path + "haarcascade_frontalface.xml"
faceCascade = cv.CascadeClassifier(cascPath)

def display_rgb(dev, data, timestamp):
    global keep_running,tilt_angle,r_color,picture_flag, data_path
    #Capturing frame by frame
    frame=frame_convert.video_cv(data)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Face detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    height, width, depth = frame.shape
    r_area = (180,180)
    rx = (width-r_area[0])/2
    ry = (height-r_area[1])/2
    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #Uncomment to watch detection rectangle
        #cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)
        if x>=rx and x+w<=rx+r_area[0] and y>=ry and y+h<=ry+r_area[1]:
            r_color=(0,255,0)
            picture_flag=True
        else:
            r_color=(0,0,255)
            picture_flag=False
    #Draw fixed rectangle in image
    cv.rectangle(frame,(rx,ry),(rx+r_area[0],ry+r_area[1]),r_color,2)
    #Displaying the resulting frame
    cv.imshow('RGB', frame)
    #Key event handling + manual motor movement
    key = cv.waitKey(1)
    if key== 32 and picture_flag:
        print "WHISKY!"
        picture = frame[y:y+h , x:x+w]
        cv.imwrite(data_path+"test.png",picture)
    if key == 65362: #Up key
        if not tilt_angle+3>27:
            tilt_angle+=3
    if key== 65364: #Down key
        if not tilt_angle-3<-27:
            tilt_angle-=3
    if key == ord('q'):
        keep_running = False

def body(dev,ctx):
    global tilt_angle
    #Initializes tilt_angle variable for motor control
    if tilt_angle==None:
        freenect.update_tilt_state(dev)
        tilt_angle=freenect.get_tilt_degs(freenect.get_tilt_state(dev))
    #If the movement variable is updated, moves the motor
    if not freenect.get_tilt_degs(freenect.get_tilt_state(dev))==tilt_angle:        
        freenect.set_tilt_degs(dev, tilt_angle)
        freenect.update_tilt_state(dev)
    #Exit condition for finishing execution
    if not keep_running:
        cv.destroyAllWindows()
        raise freenect.Kill()

freenect.runloop(video=display_rgb,
                 body=body)
