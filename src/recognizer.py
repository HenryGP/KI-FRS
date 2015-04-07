""" Module focused on face recognition """
import numpy as np
import cv2 as cv
import os.path
import glob
from numpy.oldnumeric.linear_algebra import eigenvalues
from file_manager import Picture_Manager

"""
    Main face recognition class, manager all the methods covered
    type: 1 -> eigenfaces, 2 -> fisherfaces
    source: bw, nmtx, depth
"""
class Recognizer():
    manager = Picture_Manager()
    
    """Initialization of model object through manager functionality"""
    def __init__(self,type,source,num_components=3,threshold=100.0):
        self.type = type
        self.model = self.manager.load_model(self.type,source,num_components,threshold)
    
    """Training model method"""
    def tr(self,source="bw"):
        tr,labels,names=self.manager.get_samples("tr",source)
        self.model.train(tr,labels)
        self.manager.save_model(self.type,source,self.model)
        return
    
    """Test model method, returns error percentage"""
    def ts(self,source="bw"):
        ts,labels,names = self.manager.get_samples("ts",source)
        images = ts.shape[0]; error = 0
        for img in xrange(ts.shape[0]):
            result,confidence=self.model.predict(ts[img])
            """print "======================================="
            print "Result: ",result," LABEL: ",labels[img][0]
            print "Confidence: ", confidence
            print "=======================================
            """
            if not (result==int(labels[img][0])):
                error +=1
        return (float(error)/images)*100.0
        
"""Recognizer(1,"nmtx").tr("nmtx")
ret = Recognizer(1,"nmtx").ts("nmtx")
print "Total error: ",ret"""