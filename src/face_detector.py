"""
    Face detection module.
    Based on OpenCV sample.
    Code at:
    https://github.com/Itseez/opencv/blob/master/samples/python2/facedetect.py
    Enhancements being done on other branch
"""

import freenect
import cv2 as cv
import frame_convert
import os.path

#Global variables
keep_running = True
tilt_angle=None
#Window creation
cv.namedWindow('RGB')

#Path to .xml file
cascPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +"/data/" + "haarcascade_frontalface.xml"
faceCascade = cv.CascadeClassifier(cascPath)

def display_rgb(dev, data, timestamp):
    global keep_running,tilt_angle
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
    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)
    #Displaying the resulting frame
    cv.imshow('RGB', frame)
    #Key event handling + manual motor movement
    key = cv.waitKey(1)
    if key == 65362: #Up key
        if not tilt_angle+3>27:
            tilt_angle+=3
    if key== 65364: #Down key
        if not tilt_angle-3<-27:
            tilt_angle-=3
    if key == ord('q'):
        keep_running = False

def body(dev,ctx):
    global last_time,tilt_angle
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