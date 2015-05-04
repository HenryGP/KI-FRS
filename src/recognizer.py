""" Module focused on face recognition """
import numpy as np
import cv2 as cv
import os.path
import glob
from itertools import chain
from numpy.oldnumeric.linear_algebra import eigenvalues
from file_manager import Picture_Manager
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

"""
    Main face recognition class, manager all the methods covered
    type: 1 -> eigenfaces, 2 -> fisherfaces
    source: bw, nmtx, depth
"""
class Recognizer():
    manager = Picture_Manager()
    types = {1: "Eigenfaces",2: "Fisherfaces"}
    sources = {"bw":"Raw B&W","depth":"Raw depth","nbw":"Normalized B&W","ndepth":"Normalized depth"}
    num_components = {"nbw":[41,13],"ndepth":[17,13]}
    source=None
    
    """Initialization of model object through manager functionality"""
    def __init__(self,type,source=None,num_components=None):
        self.type = type
        if source!=None:
            self.source = source
            self.model = self.manager.load_model(self.type,self.source,self.num_components[self.source][self.type-1])
    
    def predict(self,mode,bw_img,depth_img):
        bw_model = self.manager.load_model(self.type,"nbw",self.num_components["nbw"][self.type-1])        
        bw_prediction,bw_confidence = bw_model.predict(bw_img)
        print "BW prediction: ",bw_prediction," ",bw_confidence
        bw_model = self.manager.load_model(self.type,"ndepth",self.num_components["ndepth"][self.type-1])
        depth_prediction,depth_confidence = bw_model.predict(depth_img)
        print "Depth prediction: ",depth_prediction," ",depth_confidence
    
    """Training model method"""
    def tr(self,source):
        if self.source==None:
            self.source=source
            self.source=self.manager.load_model(self.type,self.source,num_components)
        tr,labels,names=self.manager.get_samples("tr",source)
        self.model.train(tr,labels)
        self.manager.save_model(self.type,source,self.model)
        return
    
    """Test model method, returns error percentage"""
    def ts(self,source="bw"):
        ts,labels,names = self.manager.get_samples("ts",source)
        images = ts.shape[0]; error = 0
        x=list(set(list(chain.from_iterable(labels.tolist()))))
        y = [0] * len(x)
        for img in xrange(ts.shape[0]):
            result,confidence=self.model.predict(ts[img])
            print "======================================="
            print "Result: %s LABEL: %s" %(result,labels[img][0])
            print "Confidence: %.2f" % (confidence)
            print "======================================="
            
            if not (result==int(labels[img][0])):
                y[x.index(labels[img][0])]+=1
                error +=1
        #self.plot_results(x, y,(float(error)/images)*100.0)
        return (float(error)/images)*100.0
    
    def plot_results(self,x,y,error):
        N = len(x)
        ind = np.arange(N)
        plt.bar(x,y,1,color='r')
        title = "Comparative graphic %s \n %s"%(self.types[self.type],self.sources[self.source])
        plt.title(title)
        legend = "Error: %.2f%s"%(error,"%")
        plt.text(max(x)/2., max(y)+.5, legend)
        plt.xlabel("Subjects")
        plt.ylabel("Fails")
        plt.xticks(ind+1/2.,np.arange(0,len(x),1))
        plt.axis([1,len(x)+1,0,max(y)])
        plt.yticks(np.arange(0,max(y)+2,1))
        plt.show()
               
"""
rec = Recognizer(1,"nbw",43)
rec.tr("nbw")
ret = rec.ts("nbw")
"""

#Test for eigenfaces to check recognition precision
def test_components(type,source,min,max):
    x = []; y=[]
    for i in range(min,max+1):
        rec=Recognizer(type,source,num_components=i)
        rec.tr(source)
        ret = rec.ts(source)
        x.append(i); y.append(ret)
    plt.plot(x,y)
    plt.title("Components adjustment")
    plt.xlabel("No. Components")
    plt.ylabel("Error %")
    plt.show()


"""Code taken from repository: 
https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py#L124"""
def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def model_testing(type,source):
    if source == "nbw" and type==1:
        comps = 43
        shape = (100,100)
    elif source == "ndepth" and type==1:
        comps = 37
        shape = (100,100)
    elif source == "bw" and type==1:
        comps = 43
        shape = (192,256)
    elif source=="depth"  and type==1:
        comps = 84
        shape = (192,256)
    elif source == "nbw" and type==2:
        comps = 13
        shape = (100,100)
    elif source == "ndepth" and type==2:
        comps = 13
        shape = (100,100)
    elif source == "bw" and type==2:
        comps = 13
        shape = (192,256)
    elif source=="depth"  and type==2:
        comps = 15
        shape = (192,256)
    rec = Recognizer(type,source,comps)
    rec.tr(source)
    mean = rec.model.getMat("mean")
    eigenvectors = rec.model.getMat("eigenvectors")
    mean_norm = normalize(mean, 0, 255, dtype=np.uint8)
    mean_resized = mean_norm.reshape(shape)
    #cv.imshow("mean", mean_resized)
    cv.imwrite("mean.png", mean_resized)
    for i in xrange(16):
        eigenvector_i = eigenvectors[:,i].reshape(shape)
        eigenvector_i_norm = normalize(eigenvector_i, 0, 255, dtype=np.uint8)
        #cv.imshow("eigenface_%d" % (i), eigenvector_i_norm)
        cv.imwrite("eigenface_%d.png" % (i), eigenvector_i_norm)
    cv.waitKey()
    














    