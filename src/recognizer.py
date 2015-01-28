"""
    Module focused on face recognition
"""

import numpy as np
import cv2 as cv
import os.path
import glob
from numpy.oldnumeric.linear_algebra import eigenvalues
data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +"/data/"

def pca(mode="tr"):
    global data_path
    #model = cv.createEigenFaceRecognizer(num_components=10,threshold=100.0)
    def pca_tr(path):
        pass
            
    def pca_ts(path):
        """"
        matrix_test = None
        for img in glob.glob1(path,'*_bw.png'):
            bw_img = cv.imread(path+img,0)
            img_vector = bw_img.reshape(bw_img.shape[0]*bw_img.shape[1])
            try:
                matrix_test = np.vstack((matrix_test,img_vector))
            except:
                matrix_test = img_vector
        mean, eigenvectors = cv.PCACompute(matrix_test, np.mean(matrix_test, axis=0).reshape(1,-1))
        """
    if mode == "tr":
        return pca_tr(data_path + 'tr/')
    else:
        return pca_ts(data_path + 'ts/')

pca()







