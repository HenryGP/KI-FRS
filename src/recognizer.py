"""
    Module focused on face recognition
"""

import numpy as np
import cv2 as cv
import os.path
import glob
from numpy.oldnumeric.linear_algebra import eigenvalues
from file_manager import Picture_Manager

"""Main face recognition class, manager all the methods covered"""
class Recognizer():
    manager = Picture_Manager()
    
    """Initialization of model object through manager functionality"""
    def __init__(self,type=1,num_components=5,threshold=100.0):
        self.type = type
        self.model = self.manager.load_model(type,num_components,threshold)
    
    """Training model method"""
    def tr(self,type="bw"):
        tr,labels=self.manager.get_samples("tr",type)
        self.model.train(tr,labels)
        self.manager.save_model(self.type,self.model)
    
    """Test model method, returns error percentage"""
    def ts(self,type="bw"):
        ts,labels = self.manager.get_samples("ts",type)
        images = ts.shape[0]; error = 0
        for img in xrange(ts.shape[0]):
            result,confidence=self.model.predict(ts[img])
            #print "RESULT: ",result
            #print "LABEL: ",labels[img]data
            if not result-labels[img]==0:
                error +=1
        return (error/images)*100
        
test = Recognizer(1)
test.tr(type="bw")
print "Classification error: ", test.ts(type="bw")