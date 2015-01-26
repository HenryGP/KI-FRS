"""
    Interface to the Kinect device, provides services for fetching images in RGB and depth
    straight from the sensor.
    Intended for sample taking for training purposes.
"""
import freenect
import numpy as np
import cv2 as cv
import os.path

"""
    State file data, saves tr and ts number of images
"""
state_file = None
tr_counter = 0
ts_counter = 0

"""
    Kinect global variables
"""
keep_running = True
tilt_angle=None
picture_flag = False
run_mode = None

"""
    Global variables for Depth, RGB & BW images
"""
bw_img = None
depth_img = None
rgb_img = None

"""
    Fixed rectangle properties (detection assistant)
"""
r_color = (0,0,255)
r_area = (192,256)
rx = None
ry = None

"""Path to the multiple data being managed"""
#Data path
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +"/data/"
#Detector path
detect_path = data_path + "detector/"
#Training path
tr_path = data_path + "tr/"
#Test path
ts_path = data_path + "ts/"
#.xml file with cascade detector for frontal faces
faceCascade = cv.CascadeClassifier(detect_path + "haarcascade_frontalface.xml")

"""Windows for image display"""
#Window creation
cv.namedWindow('RGB')
#cv.namedWindow('Depth') #If depth image needs to be displayed

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
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def file_saving():
    """
        Saves the files taken from the sensor to it's corresponding directory
        according to run_mode
    """
    global rx,ry,r_area, ts_counter, tr_counter
    print "SMILE!" #Polite salutation, useful to realize when the photo is taken
    if run_mode == "tr":
        tr_counter+=1
        cv.imwrite(tr_path+str(tr_counter)+"_bw.png",bw_img[ry:ry+r_area[1],rx:rx+r_area[0]])
        cv.imwrite(tr_path+str(tr_counter)+"_depth.png",depth_img[ry:ry+r_area[1],rx:rx+r_area[0]])
        cv.imwrite(tr_path+str(tr_counter)+"_rgb.png",rgb_img[ry:ry+r_area[1],rx:rx+r_area[0]])
    else:
        ts_counter+=1
        cv.imwrite(ts_path+str(ts_counter)+"_bw.png",bw_img[ry:ry+r_area[1],rx:rx+r_area[0]])
        cv.imwrite(ts_path+str(ts_counter)+"_depth.png",depth_img[ry:ry+r_area[1],rx:rx+r_area[0]])
        cv.imwrite(ts_path+str(ts_counter)+"_rgb.png",rgb_img[ry:ry+r_area[1],rx:rx+r_area[0]])

def keyboard_handler():
    """
        Handler to manage keyboard interaction to move sensor tilt angle, end execution and
        pull data.
    """
    global keep_running, tilt_angle 
    key = cv.waitKey(1)
    if key== 32: #File saving
        file_saving()
    if key == 65362: #Up key
        if not tilt_angle+3>27:
            tilt_angle+=3
    if key== 65364: #Down key
        if not tilt_angle-3<-27:
            tilt_angle-=3
    if key == ord('q') or key==27: #End execution
        keep_running = False

def display_rgb(dev, data, timestamp):
    """
        Displays the RGB image for sampling
    """
    global bw_img, rgb_img, picture_flag, r_color, r_area,rx,ry
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
    cv.imshow('RGB', tmp)
    #Key event handling + manual motor movement
    keyboard_handler()

def display_depth(dev,data,timestamp):
    """
        Depth image update, is also possible to display the image into a window
        by uncommenting corresponding sections
    """
    global depth_img
    #Capturing frame by frame
    depth_img=depth_cv(data)
    #Displaying the resulting frame
    #cv.imshow('Depth', depth_img)

def body(dev,ctx):
    """
        Execution body for the runloop, manages execution init for Kinect-related parameters.
        Terminates all running processes to safely shutdown device services.
    """
    global tilt_angle,state_file,tr_counter
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
        #Writes state file
        state_file.write("tr: "+str(tr_counter)+"\n"+"ts: "+str(ts_counter))
        state_file.close()
        cv.destroyAllWindows() # Kills all windows
        raise freenect.Kill() # Shutdowns all Kinect's services

def init():
    """
        Initializes counter for training and test files by reading
        the data through the state file.
    """
    global state_file
    state_file = open(data_path + "st","w+b")
    lines = state_file.readlines()
    for line in lines:
        if line.startswith("tr"):
            tmp=line.split(" ")
            tr_counter = tmp[1]
        if line.startswith("ts"):
            tmp=line.split(" ")
            ts_counter = tmp[1]

def start(mode = "tr"):
    """
        Start method for the module, used by other modules to initialize
        Kinect services.
    """
    global run_mode 
    run_mode= mode
    init() #Tmp location of method, in the future will be removed
    freenect.runloop(video=display_rgb,depth=display_depth,body=body)
    
start()