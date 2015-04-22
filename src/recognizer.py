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
    
    """Initialization of model object through manager functionality"""
    def __init__(self,type,source,num_components=3):
        self.type = type
        self.source = source
        self.model = self.manager.load_model(self.type,source,num_components)
        
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
        x=list(set(list(chain.from_iterable(labels.tolist()))))
        y = [0] * len(x)
        for img in xrange(ts.shape[0]):
            result,confidence=self.model.predict(ts[img])
            """print "======================================="
            print "Result: ",result," LABEL: ",labels[img][0]
            print "Confidence: ", confidence
            print "======================================="
            """
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

