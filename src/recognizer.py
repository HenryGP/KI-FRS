""" 
    Module focused on face recognition 
    Includes:
        - Recognizer class: specialized for the KI-FRS
            - Auxiliary options for graphics
        - External DB Recognizer: modified version to use with external databases
"""
import numpy as np
import cv2 as cv
import os.path
import glob
from itertools import chain
from numpy.oldnumeric.linear_algebra import eigenvalues
from file_manager import Picture_Manager
from file_manager import External_DB_Manager
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import preprocessor

"""
    Main face recognition class, manager all the methods covered
    type: 1 -> eigenfaces, 2 -> fisherfaces
    source: bw, nmtx, depth
"""
class Recognizer():
    manager = Picture_Manager()
    types = {1: "Eigenfaces",2: "Fisherfaces"}
    sources = {"bw":"Raw B&W","depth":"Raw depth","nbw":"Normalized B&W","ndepth":"Normalized depth"}
    source=None
    
    """Initialization of model object through manager functionality"""
    def __init__(self,type,source=None,num_components=None):
        self.type = type
        if not source==None:
            self.source = source
            self.model = self.manager.load_model(self.type,self.source)
            if self.model == None:
                if type==1:
                    maxi = 100
                else:
                    maxi = 16
                self.model = self.manager.load_model(self.type,self.source,self.optimum_components(type,source,5,maxi,False))
    
    """Given a black&white and depth image, according to the selected mode, predicts the label"""               
    def predict(self,mode,bw_img,depth_img):
        if mode=="auto":
            bw_model = self.manager.load_model(self.type,"nbw")        
            bw_prediction,bw_confidence = bw_model.predict(bw_img)
            print "================================="
            print "BW prediction: ",bw_prediction," ",bw_confidence
            print "================================="
            depth_model = self.manager.load_model(self.type,"ndepth")
            depth_prediction,depth_confidence = depth_model.predict(depth_img)
            print "================================="
            print "Depth prediction: ",depth_prediction," ",depth_confidence
            print "================================="
            name,rgb_img,depth_img = self.manager.get_sample_info(bw_prediction)
            cv.imshow(name,rgb_img)
            name,rgb_img,depth_img = self.manager.get_sample_info(depth_prediction)
        elif mode=="nbw":
            bw_model = self.manager.load_model(self.type,"nbw")        
            bw_prediction,bw_confidence = bw_model.predict(bw_img)
            print "================================="
            print "BW prediction: ",bw_prediction," ",bw_confidence
            print "================================="
            name,rgb_img,depth_img = self.manager.get_sample_info(bw_prediction)
            cv.imshow(name,rgb_img)
        else:
            depth_model = self.manager.load_model(self.type,"ndepth")
            depth_prediction,depth_confidence = depth_model.predict(depth_img)
            print "================================="
            print "Depth prediction: ",depth_prediction," ",depth_confidence
            print "================================="
            name,rgb_img,depth_img = self.manager.get_sample_info(depth_prediction)
            cv.imshow(name,rgb_img)
        cv.waitKey(0)
    
    """Training model method"""
    def tr(self,save=True):
        tr,labels,names=self.manager.get_samples("tr",self.source)
        self.model.train(tr,labels)
        if save:
            self.manager.save_model(self.type,self.source,self.model)
        return
    
    """Test model method, returns error percentage. Plot possibility available calling plot_results function"""
    def ts(self,plot=False):
        ts,labels,names = self.manager.get_samples("ts",self.source)
        images = ts.shape[0]; error = 0
        x=list(set(list(chain.from_iterable(labels.tolist()))))
        y = [0] * len(x)
        for img in xrange(ts.shape[0]):
            result,confidence=self.model.predict(ts[img])
            """print "======================================="
            print "Result: %s LABEL: %s" %(result,labels[img][0])
            print "Confidence: %.2f" % (confidence)
            print "======================================="""        
            if not (result==int(labels[img][0])):
                y[x.index(labels[img][0])]+=1
                error +=1
        if plot:
            self.plot_results(x, y,(float(error)/images)*100.0)
        return (float(error)/images)*100.0
    
    """Called from the ts function, creates an historam counting the number of false positives for each label"""
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
    
    """Creates a test model, used for component adjustment"""
    def create_model(self,type,source,num_components):
        if type==1:    
            self.model = cv.createEigenFaceRecognizer(num_components)
        elif type==2:
            self.model = cv.createFisherFaceRecognizer(num_components)
    
    """Calculates the optimum number of components to preserve given a certain range"""       
    def optimum_components(self,type,source,mini,maxi,plot=False):
        x = []; y=[]
        for i in range(mini,maxi+1):
            self.create_model(type, source, i)
            self.tr(False)
            ret = self.ts()
            x.append(i); y.append(ret)
        if plot==True:
            plt.plot(x,y)
            plt.title("Components adjustment")
            plt.xlabel("No. Components")
            plt.ylabel("Error %")
            plt.show()
        return x[y.index(min(y))]
       
    """Code taken from repository: 
    https://github.com/Itseez/opencv/blob/2.4/samples/python2/facerec_demo.py#L124"""
    def normalize(self,X, low, high, dtype=None):
        """Normalizes a given array in X to a value between low and high."""
        X = np.asarray(X)
        minX, maxX = np.min(X), np.max(X)
        # normalize to [0...1].
        X = X - float(minX)
        X = X / float((maxX - minX))
        # scale to [low...high].normalize
        X = X * (high-low)
        X = X + low
        if dtype is None:
            return np.asarray(X)
        return np.asarray(X, dtype=dtype) 

    """Displays the number of eigenvectors specified and the mean face"""       
    def show_eigenvectors(self,num_components):
        mean = self.model.getMat("mean")
        eigenvectors = rec.model.getMat("eigenvectors")
        mean_norm = self.normalize(mean, 0, 255, dtype=np.uint8)
        mean_resized = mean_norm.reshape((100,100))
        cv.imwrite("mean.png", mean_resized)
        #cv.imshow("mean face", mean_resized)
        for i in xrange(num_components+1):
            eigenvector_i = eigenvectors[:,i].reshape((100,100))
            eigenvector_i_norm = self.normalize(eigenvector_i, 0, 255, dtype=np.uint8)
            cv.imshow("eigenface_%d" % (i), eigenvector_i_norm)
            #cv.imwrite("eigenface_%d.png" % (i), eigenvector_i_norm)
        cv.waitKey()

"""
rec = Recognizer(2,"ndepth")
rec.tr()
ret = rec.ts(True)
print "Percentage: %f"%(ret)
"""     

"""Specific class to manage external databases testing"""   
class ExternalDB_Recognizer(Recognizer):
    """Initialization of a model given a partition percentage to create the training and testing sets"""
    def __init__(self,type,database,partition_percentage,num_components=None):
        if not num_components==None:
            if type==1:
                self.model = cv.createEigenFaceRecognizer(num_components)
            else:
                self.model = cv.createFisherFaceRecognizer(num_components) 
        else:
            if type==1:
                self.model = cv.createEigenFaceRecognizer()
            else:
                self.model = cv.createFisherFaceRecognizer()
        self.manager = External_DB_Manager(partition_percentage)
        self.database = database
    
    """For benchmarking the algorithm performance, first trains the model and then tests it"""
    def tr_and_ts(self,filter=True):
        tr_samples,tr_labels,ts_samples,ts_labels = self.manager.get_data(self.database) 
        self.tr(tr_samples,tr_labels,filter)
        return self.ts(ts_samples,ts_labels,filter)
    
    """Training the model with the given training samples and labels, has the possibility of filtering the set"""
    def tr(self,tr_samples,tr_labels,filter=True):
        print "Training model of %s raw samples"%(str(len(tr_samples)))
        samples = []; labels =[]; accepted=0
        for i in xrange(len(tr_samples)):
            sample = cv.imread(tr_samples[i],0)
            if (not sample==None) and (filter) :
                sample, d1, d2 = preprocessor.normalize_sample(sample,yale=(self.database=="yale"))
            elif not sample==None:    
                sample = cv.resize(sample,(100,100))
            if not sample==None:
                accepted+=1
                try:    
                    samples = np.vstack((samples,sample.reshape(sample.shape[0]*sample.shape[0])))
                    labels = np.vstack((labels,tr_labels[i]))
                except:
                    samples = sample.reshape(sample.shape[0]*sample.shape[0])
                    labels = tr_labels[i]        
        print "Tr processed samples: ",accepted
        self.model.train(samples,labels)
    
    """Tests the trained model. The testing set could be also filtered or not by the preprocessor"""
    def ts(self,ts_samples,ts_labels,filter=True):
        print "Ts raw samples: ",len(ts_samples)
        error = 0; accepted = 0
        for i in xrange(len(ts_samples)):
            sample = cv.imread(ts_samples[i],0)
            if (not sample==None) and (filter):
                sample, d1, d2 = preprocessor.normalize_sample(sample,yale=(self.database=="yale"))
            elif not sample==None:
                sample = cv.resize(sample,(100,100))
            if not sample==None:
                accepted += 1
                result,confidence=self.model.predict(sample.reshape(sample.shape[0]*sample.shape[0]))
                """print "======================================="
                print "Result: %s LABEL: %s" %(result,ts_labels[i])
                print "Confidence: %.2f" % (confidence)
                print "======================================="""
                if not result==ts_labels[i]:
                    error +=1
        print "Ts processed samples: ",accepted
        print "Hit rate: ",100.0-(float(error)/accepted)*100.0
        return 100.0-(float(error)/accepted)*100.0


#rec = ExternalDB_Recognizer(2,"yale",70)
#rec.tr_and_ts(False)


  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
    
    