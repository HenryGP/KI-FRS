"""
    Pre-processor for images
    Includes:
    - Light-related algorithms
    - Depth normalization algorithms
    - Auxiliaries
"""

import math
import numpy as np
from file_manager import Picture_Manager
import cv2 as cv

manager = Picture_Manager()

"""
    Estimates light intensity in given RGB image
    Based on the estimated value, processes the BW image
"""
def light_estimation(rgb_img):
    V = cv.cvtColor(rgb_img,cv.COLOR_RGB2HSV)[:,:,2]
    print "Mean value: ",cv.mean(V)[0]

"""
    Applies retina modeling algorithm to BW image
"""
def retina_processing(bw_img):
    pass

"""
   Gets all depth matrixes taken from the samples directory and normalizes them
"""
def normalize_depth(mode="tr"):
    samples,labels=manager.get_samples(mode,"mtx")
    if not len(samples)==0:
        samples = samples.astype(np.float32)
        for i in xrange(samples.shape[0]):
            matrix = samples[i]
            for p in xrange(matrix.shape[0]):
                matrix[p]=0.1236 * math.tan(float(matrix[p]) / 2842.5 + 1.1863) * 100 #in centimeters
            samples[i]=matrix
        manager.save_samples(mode,"mtx",samples,labels) 
    return len(samples)