"""
    Pre-processor for images
    Includes:
    - Light-related algorithms: light estimation and normalization
    - Depth normalization algorithms: matrix and depth image.
    - Auxiliaries
"""

import math
import numpy as np
from file_manager import Picture_Manager
from file_manager import Sample_Manager
from file_manager import External_DB_Manager
import cv2 as cv

manager = Picture_Manager()
bw_detector = Sample_Manager().faceCascade
depth_detector = Picture_Manager().faceCascade

def light_estimation(img):
    """
        Estimates light intensity in given RGB image
        Based on the estimated value, processes the BW image
        Returns True if the light estimation is OK, False in the other case
    """
    V = cv.cvtColor(img,cv.COLOR_RGB2HSV)[:,:,2]
    # Under 80 is too dark, needs to be processed
    if cv.mean(V)[0] < 80.0: 
        return False
    return True 

def normalize_depth_map(mode="tr"):
    """
       Gets all depth matrixes taken from the samples directory and normalizes them
    """
    samples,labels, names=manager.get_samples(mode,"mtx")
    if not len(samples)==0:
        samples = samples.astype(np.float32)
        for i in xrange(samples.shape[0]):
            matrix = samples[i]
            for p in xrange(matrix.shape[0]):
                matrix[p]=0.1236 * math.tan(float(matrix[p]) / 2842.5 + 1.1863) * 100 #in centimeters
            samples[i]=matrix
        manager.save_samples(mode,"mtx",samples,labels) 
    return len(samples)

def cut_image(bw_img,depth_img=None,depth_mtx=None):
    """
        Using face detectors, cuts the B&W and depth images.
        As depth face detector is inaccurate, bases the cutting pixels on the result of the B&W detector
        THe last returned parameter indicates if the sample is suitable of not for using in the Recognizer
    """
    global counter
    result_bw=bw_img;result_mtx=depth_mtx;result_depth=depth_img;
    if not depth_img==None:
        # Face detection on depth image
        faces = depth_detector.detectMultiScale(
            depth_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        if len(faces)==0: #Only considers if at least one face was detected
            return result_bw, result_depth,result_mtx, True
    faces = bw_detector.detectMultiScale(
        bw_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        result_bw = bw_img[y:y+h,x:x+w]
        if not depth_img==None:
            result_depth = depth_img[y:y+h,x:x+w]
            result_mtx = depth_mtx[y:y+h,x:x+w]
    if len(faces)==0:
        return result_bw, result_depth,result_mtx, True
    else:
        return result_bw, result_depth,result_mtx, False  

def light_normalization(img):
    """
        Normalizes light conditions in the given B&W image
    """
    #Histogram equalization
    img = cv.equalizeHist(img)
    #Gamma correction with factor 0.8 (smaller factors -> more bright)
    img = img/255.0
    img = cv.pow(img,0.8)
    img = np.uint8(img*255)
    img = cv.fastNlMeansDenoising(img,10,10,7,21)
    #Gaussian filter to smooth
    img = cv.GaussianBlur(img,(3,3),0)
    return img

def normalize_depth_image(depth_img,map_mtx):
    """Normalizes the depth image by using a median blur filter"""
    return cv.medianBlur(depth_img,5)

def normalize_images(mode="tr"):
    """
        Normalizes all data related to either training or testing, this includes depth, B&W and raw matrixes.
        The result is a 100x100 image of the face.
    """
    #RGB images for light estimation
    rgb_imgs, rgb_labels, rgb_names = manager.get_samples(mode,"rgb")
    #Raw depth images
    depth_imgs,depth_labels,depth_names = manager.get_samples(mode,"depth")
    #Depth map
    map_mtxs,map_labels,map_names = manager.get_samples(mode,"mtx")
    #Resulting matrixes of the preprocessing function
    bw_new_imgs = []; depth_new_imgs = []
    names_dict = dict((el,[]) for el in rgb_labels)
    for idx in xrange(len(rgb_names)):
        names_dict[rgb_labels[idx]].append((rgb_names[idx][0:rgb_names[idx].find("_")],idx))
    rgb_new_labels = []; rgb_new_names = []
    for idx in xrange(len(rgb_imgs)):
        #print "Analysis of rgb image: ",rgb_names[idx]," , ",rgb_labels[idx]
        #Preprocessing of the BW image
        if light_estimation(rgb_imgs[idx])==False:
            bw_img = manager.get_sample(mode,"bw",rgb_labels[idx],rgb_names[idx]).reshape(256,192)
            bw_img = light_normalization(bw_img)
        else:
            bw_img = manager.get_sample(mode,"bw",rgb_labels[idx],rgb_names[idx]).reshape(256,192)
        ##############################
        #Preprocessing of corresponding depth image
        depth_img_name = rgb_names[idx][0:rgb_names[idx].find("_")]+"_depth.png"
        map_mtx_name = rgb_names[idx][0:rgb_names[idx].find("_")]+"_mtx.npy"
        depth_img = manager.get_sample(mode,"depth",rgb_labels[idx],depth_img_name).reshape(256,192)
        map_mtx = manager.get_sample(mode,"nmtx",rgb_labels[idx],map_mtx_name).reshape(256,192)
        r_bw,r_depth,r_mtx,rejected = cut_image(bw_img,depth_img,map_mtx)
        if rejected:
            continue
        r_depth = normalize_depth_image(r_depth,r_mtx)
        ##############################
        #Resizing normalization for images
        r_bw = cv.resize(r_bw,(100,100));r_bw = r_bw.reshape(r_bw.shape[0]*r_bw.shape[1])       
        r_depth = cv.resize(r_depth,(100,100));r_depth = r_depth.reshape(r_depth.shape[0]*r_depth.shape[1])
        ##############################
        try:
            bw_new_imgs = np.vstack((bw_new_imgs,r_bw))
            depth_new_imgs = np.vstack((depth_new_imgs,r_depth))
        except:
            bw_new_imgs = r_bw
            depth_new_imgs = r_depth
        rgb_new_labels.append(rgb_labels[idx])
        rgb_new_names.append(rgb_names[idx])
    manager.save_samples(mode,"bw",bw_new_imgs,rgb_new_labels,rgb_new_names)
    manager.save_samples(mode,"depth",depth_new_imgs,rgb_new_labels,rgb_new_names)
    return len(rgb_new_labels)

def normalize_sample(rgb_img,depth_img=None,depth_mtx=None,yale=False):
    """
        Given an individual sample, normalizes it by estimating light conditions
        and later cutting and scaling it. The result is a 100x100 image of the face.
    """
    if not len(rgb_img.shape)==3:
        bw_img = rgb_img
        rgb_img = cv.cvtColor(rgb_img,cv.COLOR_GRAY2RGB)
    else:
        bw_img = cv.cvtColor(rgb_img,cv.COLOR_RGB2GRAY)
    if light_estimation(rgb_img)==False:
        bw_img = light_normalization(bw_img)        
    if not yale:
        r_bw,r_depth,r_mtx,rejected = cut_image(bw_img,depth_img,depth_mtx)
    else:
        r_bw = bw_img;r_depth=None;r_mtx=None;rejected=False
    if rejected:
        return None,None,None
    r_bw = cv.resize(r_bw,(100,100))
    if not depth_img==None:
        r_depth = normalize_depth_image(r_depth,r_mtx)
        r_depth = cv.resize(r_depth,(100,100))
    return r_bw,r_depth,r_mtx
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

