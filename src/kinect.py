"""
    Interface to the Kinect device, provides services for fetching images in RGB and depth
    straight from the sensor.
    Intended for sample taking for training purposes.
"""
import freenect
import numpy as np
import cv2 as cv
import os.path
from file_manager import Sample_Manager

"""
    Kinect global variables
"""
keep_running = True
tilt_angle=None
picture_flag = False
run_mode = None

"""
    Global variables for Depth, RGB ,BW images & Depth Matrix for pre-processing 
"""
bw_img = None
depth_img = None
rgb_img = None
depth_mtx = None

"""
    Fixed rectangle properties (detection assistant)
"""
r_color = (0,0,255)
r_area = (192,256)
rx = None
ry = None

"""File manager object"""
file_manager = None
faceCascade = None

"""Windows for image display"""
#Window creation
cv.namedWindow('Viewer')
#cv.namedWindow('Depth') #If depth image needs to be displayed

"""Depth image allignment"""
K = np.array([[385.58130632872212, 0, 371.50000000000000], [0, 385.58130632872212, 236.50000000000000], [0, 0, 1]])
d = np.array([-0.27057628187805571, 0.10522881965331317, 0, 0, 0]) # just use first two terms (no translation)
newcamera, roi = cv.getOptimalNewCameraMatrix(K, d, (640,480), 0)

def video_cv(video):
    """Converts video into a BGR format for opencv
    This is abstracted out to allow for experimentation
    Args:
        video: A numpy array with 1 byte per pixel, 3 channels RGB
    Returns:
        An opencv image who's datatype is 1 byte, 3 channel BGR
    """
    video = video[:, :, ::-1]  # RGB -> BGR
    return video.astype(np.uint8)

def depth_cv(depth):
    """Converts depth into a 'nicer' format for display
        This is abstracted to allow for experimentation with normalization
        Args:
            depth: A numpy array with 2 bytes per pixel
        Returns:
            A numpy array that has been processed whos datatype is unspecified
    """
    np.clip(depth, 0,(2**11)-1, depth)
    mtx = np.matrix.copy(depth)
    #depth >>= 2
    depth = depth.astype(np.uint8)
    return depth,mtx

def get_sample():
    global rx,ry,r_area, file_manager, run_mode,K,d,newcamera
    depth_align = cv.undistort(depth_img, K, d, None, newcamera)
    return rgb_img[ry:ry+r_area[1],rx:rx+r_area[0]],depth_align[ry-31:ry+r_area[1]-31,rx+31:rx+r_area[0]+31],depth_mtx


def file_saving(name):
    """
        Saves the files taken from the sensor to it's corresponding directory
        according to run_mode
    """
    global rx,ry,r_area, file_manager, run_mode,K,d,newcamera
    print "SMILE!" #Polite salutation, useful to realize when the photo is taken
    depth_align = cv.undistort(depth_img, K, d, None, newcamera)
    samples = [bw_img[ry:ry+r_area[1],rx:rx+r_area[0]],depth_align[ry-31:ry+r_area[1]-31,rx+31:rx+r_area[0]+31],rgb_img[ry:ry+r_area[1],rx:rx+r_area[0]],depth_mtx]
    file_manager.store_samples(samples,run_mode,name)

def new_sampling(mode):
    file_manager.new_sampling(mode)

def keyboard_handler():
    """
        Handler to manage keyboard interaction to move sensor tilt angle, end execution and
        pull data.
    """
    global keep_running, tilt_angle, run_mode,file_manager
    key = cv.waitKey(1)
    if key == 65362: #Up key
        if not tilt_angle+3>27:
            tilt_angle+=3
    if key== 65364: #Down key
        if not tilt_angle-3<-27:
            tilt_angle-=3

def display_rgb(dev, data, timestamp):
    """
        Displays the RGB image for sampling
    """
    global bw_img, rgb_img, picture_flag, r_color, r_area,rx,ry,faceCascade
    #Capturing frame by frame from video camera
    rgb_img=video_cv(data)
    bw_img = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY)
    #Face detection
    faces = faceCascade.detectMultiScale(
        bw_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    #Fixed rectangle parameters
    height, width, depth = rgb_img.shape 
    rx = (width-r_area[0])/2
    ry = (height-r_area[1])/2
    for (x, y, w, h) in faces:
        #Uncomment to see detection rectangle
        #cv.rectangle(rgb_img, (x, y), (x+w, y+h), (0, 255, 0),2)
        if x>=rx and x+w<=rx+r_area[0] and y>=ry and y+h<=ry+r_area[1]:
            r_color=(0,255,0)
            picture_flag=True
        else:
            r_color=(0,0,255)
            picture_flag=False 
    #Temporary image to be displayed
    tmp = rgb_img.copy()
    #Draw fixed rectangle in image
    cv.rectangle(tmp,(rx,ry),(rx+r_area[0],ry+r_area[1]),r_color,2)
    #Displaying the resulting frame
    cv.imshow('Viewer', tmp)
    #Key event handling + manual motor movement
    keyboard_handler()

def display_depth(dev,data,timestamp):
    """
        Depth image update, is also possible to display the image into a window
        by uncommenting corresponding sections
    """
    global depth_img, depth_mtx
    #Capturing frame by frame
    depth_img,depth_mtx=depth_cv(data)
    #Displaying the resulting frame
    #cv.imshow('Depth', depth_img)

def body(dev,ctx):
    """
        Execution body for the runloop, manages execution init for Kinect-related parameters.
        Terminates all running processes to safely shutdown device services.
    """
    global tilt_angle,state_file,tr_counter,file_manager
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
        del file_manager
        cv.destroyAllWindows() # Kills all windows
        raise freenect.Kill() # Shutdowns all Kinect's services
    
def start(mode = "tr"):
    """
        Start method for the module, used by other modules to initialize
        Kinect services.
    """
    global run_mode, file_manager, faceCascade
    run_mode= mode
    file_manager = Sample_Manager()
    faceCascade = file_manager.faceCascade
    freenect.runloop(video=display_rgb,depth=display_depth,body=body)
    
#start("tr")