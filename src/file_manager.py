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
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +"/data/"
    #Detector path
    detect_path = data_path + "detector/"
    #Images path
    img_path = data_path + "img/"
    #Training path
    tr_path = img_path + "tr/"
    #Test path
    ts_path = img_path + "ts/"
    #.xml file with cascade detector for frontal faces
    faceCascade = cv.CascadeClassifier(detect_path + "haarcascade_frontalface.xml")
    
    """Internal dir counters"""
    img_ptr = 0; tr_ptr = 0; ts_ptr = 0 
    
    def __init__(self,mode="tr"):
        """
            Initializes counter for training and test files by reading
            the data through the state file.
        """
        if mode=="tr": #Training mode
            self.tr_counter = sorted([x[0].split("/")[-1] for x in os.walk(self.tr_path)],reverse=True)[0]
            if self.tr_counter=="": 
                self.tr_counter=0
            else: 
                self.tr_counter=int(self.tr_counter)
            self.new_sampling("tr")
            self.tr_ptr=self.tr_counter
            self.to_dir(self.tr_ptr, "tr")
        else:
            self.ts_counter = sorted([x[0].split("/")[-1] for x in os.walk(self.ts_path)],reverse=True)[0]
            if self.ts_counter=="": 
                self.ts_counter=0
            else: 
                self.ts_counter=int(self.ts_counter)
            self.new_sampling("ts")
            self.ts_ptr=self.ts_counter
            self.to_dir(self.ts_ptr, "ts")          
            
    """def __del__(self):"""
    
    def new_sampling(self,mode="tr"):
        """Creates new directory under the dir specified by 'mode'"""
        if mode == "tr": #Creates new folder for training
            self.tr_counter += 1
            print "Path: ",self.tr_path+str(self.tr_counter)
            os.makedirs(self.tr_path+str(self.tr_counter))
            self.tr_ptr=self.tr_counter
        else: #Creates new folder for test
            self.ts_counter += 1
            print "Path: ",self.ts_path+str(self.ts_counter)
            os.makedirs(self.ts_path+str(self.ts_counter))
            self.ts_ptr=self.ts_counter
        self.img_ptr = 0
        
    def store_samples(self,samples,mode="tr"):
        """
            Gets the image samples (BW,RGB and Depth) and stores them in corresponding
            dir indicated by ptr. Images are named according to img_ptr value 
        """
        print "tr pointer: ",self.tr_ptr
        print "ts pointer: ",self.ts_ptr
        if mode=="tr": path=self.tr_path+str(self.tr_ptr)+"/"
        else: path=self.ts_path+str(self.ts_ptr)+"/"
        try:
            self.img_ptr += 1
            print "Path: ",path
            print "img_pointer: ",self.img_ptr
            cv.imwrite(path+str(self.img_ptr)+"_bw.png",samples[0])
            cv.imwrite(path+str(self.img_ptr)+"_depth.png",samples[1])
            cv.imwrite(path+str(self.img_ptr)+"_rgb.png",samples[2])
        except:
            print "Images couldn't be saved"
    
    def to_dir(self,dir_ptr,mode="tr"):
        if mode == "tr":
            if dir_ptr>self.tr_counter:
                raise
            else:
                self.tr_ptr=dir_ptr
                self.img_ptr =  [ int(f.split("_")[0]) for f in listdir(self.tr_path+str(dir_ptr)+"/") if isfile(join(self.tr_path+str(dir_ptr)+"/",f)) ]
                if len(self.img_ptr)==0:
                    self.img_ptr=0
                else:
                    self.img_ptr =  max(self.img_ptr)
                print "Tr pointer: ",self.tr_ptr
                print "Img pointer: ",self.img_ptr
        else:
            if dir_ptr>self.ts_counter:
                raise
            else:
                self.ts_ptr=dir_ptr
                self.img_ptr =  [ int(f.split("_")[0]) for f in listdir(self.ts_path+str(dir_ptr)+"/") if isfile(join(self.ts_path+str(dir_ptr)+"/",f)) ]
                if len(self.img_ptr)==0:
                    self.img_ptr=0
                else:
                    self.img_ptr =  max(self.img_ptr)
                print "Ts pointer: ",self.ts_ptr
                print "Img pointer: ",self.img_ptr

class Picture_Manager():
    """
        Class for images management for preprocessing and recognition
    """
    #Data path
    data_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +"/data/"
    #Images path
    img_path = data_path + "img/"
    #Training path
    tr_path = img_path + "tr/"
    #Test path
    ts_path = img_path + "ts/"
    #Recognition path
    rec_path = data_path + "recognition/"
    
    def get_samples(self,mode="tr",type="bw"):
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        if type == "bw":
            pattern = '*_bw.png'
        else:
            pattern = '*_depth.png'
        samples_matrix = None
        samples_labels = None
        """Building up the matrixes"""
        for label in os.listdir(path):
            for img in glob.glob1(path+str(label),pattern):
                bw_img = cv.imread(path+str(label)+"/"+str(img),0)
                img_vector = bw_img.reshape(bw_img.shape[0]*bw_img.shape[1])
                try:
                    samples_matrix = np.vstack((samples_matrix,img_vector))
                    samples_labels = np.vstack((samples_labels,int(label)))
                except:
                    samples_matrix = img_vector
                    samples_labels = int(label)
        return samples_matrix, samples_labels
        
    def load_model(self,mode,num_components,threshold):
        if mode==1:
            name = "eigenfaces.yml"
            model = cv.createEigenFaceRecognizer(num_components,threshold)
            try:
                model.load(self.rec_path+name)
            except:
                print "There was no model"
        else:
            name = "fisherfaces.yml"
            model = cv.createFisherFaceRecognizer()
            try:
                model.load(self.rec_path+name)
            except:
                print "There was no model"       
        return model
        
    def save_model(self,mode,model):
        if mode==1:
            name = "eigenfaces.yml"
        else:
            name = "fisherfaces.yml"
        model.save(self.rec_path+name)
    
#test = Picture_Manager()
#test.get_samples("tr")    
