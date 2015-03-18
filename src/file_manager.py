"""
    File Manager module to take care of data dir
"""
import cv2 as cv
import os
from os.path import *
from os import listdir
import glob
import numpy as np

class Sample_Manager():
    """
        Class for managing files taken from kinect either for tr or ts
    """
    """Path to the multiple data being managed"""
    #Data path
    data_path = "%s/data" % (os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    #Detector path
    detect_path = "%s/detector/" % (data_path)
    #Images path
    img_path = "%s/img" % (data_path)
    #Training path
    tr_path = "%s/tr/" % (img_path)
    #Test path
    ts_path = "%s/ts/" %(img_path)
    
    #.xml file with cascade detector for frontal faces
    faceCascade = cv.CascadeClassifier(detect_path + "haarcascade_frontalface.xml")
    
    def __init__(self):
        """Training set"""
        self.tr_counter = sorted([x[0].split("/")[-1] for x in os.walk(self.tr_path)])
        if not self.tr_counter[-1]=="":
            self.tr_counter = int(self.tr_counter[-1])
        else:
            self.tr_counter = 0
        """Testing set"""
        self.ts_counter = sorted([x[0].split("/")[-1] for x in os.walk(self.ts_path)])
        if not self.ts_counter[-1]=="":
            self.ts_counter = int(self.ts_counter[-1])
        else:
            self.ts_counter = 0
         
    def __del__(self):
        """Deletes directories if empty"""
        try:
            files = sorted([x[0].split("/")[-1] for x in os.walk(self.tr_path+str(self.tr_counter)+"/")])
            if len(files)==1:
                os.removedirs(self.tr_path+str(self.tr_counter))
        except:pass
        ###
        try:
            files = sorted([x[0].split("/")[-1] for x in os.walk(self.ts_path+str(self.ts_counter)+"/")])
            if len(files)==1:
                os.removedirs(self.ts_path+str(self.ts_counter))
        except: pass
        
    def new_sampling(self,mode="tr"):
        """Creates new directory under the dir specified by 'mode'"""
        if mode == "tr": #Creates new folder for training
            print "Path: ",self.tr_path+str(self.tr_counter)
            self.tr_counter+=1
            os.makedirs(self.tr_path+str(self.tr_counter))
        else: #Creates new folder for test
            print "Path: ",self.ts_path+str(self.ts_counter)
            self.ts_counter += 1
            os.makedirs(self.ts_path+str(self.ts_counter))
        self.img_ptr = 0
        
    def store_samples(self,samples,mode="tr"):
        """
            Gets the image samples (BW,RGB and Depth) and stores them in corresponding
            dir indicated by ptr. Images are named according to img_ptr value 
        """
        if mode=="tr": path=self.tr_path+str(self.tr_counter)+"/"
        else: path=self.ts_path+str(self.ts_counter)+"/"
        try:
            self.img_ptr += 1
            print "Path: ",path
            print "img_pointer: ",self.img_ptr
            cv.imwrite(path+str(self.img_ptr)+"_bw.png",samples[0])
            cv.imwrite(path+str(self.img_ptr)+"_depth.png",samples[1])
            cv.imwrite(path+str(self.img_ptr)+"_rgb.png",samples[2])
            np.save(path+str(self.img_ptr)+'_mtx.npy',samples[1])
        except:
            print "Images couldn't be saved"
    
class Picture_Manager():
    """
        Class for image management for preprocessing and recognition
    """
    #Data path
    data_path =  "%s/data/"%(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    #Images path
    img_path = "%simg/" % (data_path)
    #Training path
    tr_path = "%str/" % (img_path)
    #Test path
    ts_path = "%sts/" % (img_path)
    #Recognition path
    rec_path = "%srecognition/"%(data_path)
    
    def get_samples(self,mode="tr",type="bw"):
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        if type == "bw":
            pattern = '*_bw.png'
        elif type=="mtx":
            pattern = '*_mtx.npy'
        else:
            pattern = '*_depth.png'
        samples_matrix = None
        samples_labels = None
        """Building up the matrixes"""
        for label in os.listdir(path):
            for img in glob.glob1(path+str(label),pattern):
                if type=="bw" or type=="depth":
                    bw_img = cv.imread(path+str(label)+"/"+str(img),0)
                    img_vector = bw_img.reshape(bw_img.shape[0]*bw_img.shape[1])
                else: #Depth matrix loading
                    img_vector = np.load(path+str(label)+"/"+str(img)).reshape(192*256)
                try:
                    samples_matrix = np.vstack((samples_matrix,img_vector))
                    samples_labels = np.vstack((samples_labels,int(label)))
                except:
                    samples_matrix = img_vector
                    samples_labels = int(label)
        return samples_matrix, samples_labels

    def save_samples(self,mode,type,data,labels):
        if mode == "tr": 
            path = self.tr_path
        else: 
            path = self.ts_path
        if type == "mtx":
            c_label = -1; counter = 1
            for i in xrange(data.shape[0]):
                if labels[i][0]!= c_label:
                    c_label = labels[i][0]
                    counter=1
                mtx = data[i].reshape(256,192)
                np.savetxt(path+str(labels[i][0])+"/"+str(counter)+"_nmtx.npy",mtx)
                counter+=1

    def load_model(self,mode,num_components,threshold):
        if mode==1:
            name = "eigenfaces.yaml"
            model = cv.createEigenFaceRecognizer(num_components,threshold)
        else:
            name = "fisherfaces.yaml"
            model = cv.createFisherFaceRecognizer()
        try:
            model.load(self.rec_path+name)
        except:
            print "There was no model"
        return model
        
    def save_model(self,mode,model):
        if mode==1:
            name = "eigenfaces.yaml"
        else:
            name = "fisherfaces.yaml"
        model.save(self.rec_path+name)
    
#test = Picture_Manager()
#test.get_samples("tr")    

