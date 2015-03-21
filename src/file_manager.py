"""
    File Manager module to take care of data dir
"""
import cv2 as cv
import os
from os.path import *
from os import listdir
import glob
import numpy as np
import datetime

class Index():
    #Data path
    data_path = "%s/data/img/" % (os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        
    def __init__(self):
        self.index_file = open(self.data_path+"index.txt","a+")
        self.index_file.close()
    
    def __del__(self):
        try:
            self.index_file.close()
        except:
            pass
    
    def save_sample(self,id,name):
        if self.get_sample_name(id)==None:
            self.index_file = open(self.data_path+"index.txt","a+")
            self.index_file.write(id+";"+name+";\n")
            self.index_file.close()
    
    def get_sample_name(self,id):
        self.index_file = open(self.data_path+"index.txt","r")
        lines = self.index_file.readlines()
        self.index_file.close()
        for line in lines:
            f_id,name = line.split(";")[0],line.split(";")[1]
            if f_id==id:
                return name
        return None

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
        self.index = Index()
        """Training set"""
        self.tr_counter = sorted([int(y) for y in [x[0].split("/")[-1] for x in os.walk(self.tr_path)] if not y==""])
        if not len(self.tr_counter)==0:
            self.tr_counter = int(self.tr_counter[-1])
        else:
            self.tr_counter = 0
        
        """Testing set"""
        self.ts_counter = sorted([int(y) for y in [x[0].split("/")[-1] for x in os.walk(self.ts_path)] if not y==""])
        if not len(self.ts_counter)==0:
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
        
    def store_samples(self,samples,mode="tr",name="None"):
        """
            Gets the image samples (BW,RGB and Depth) and stores them in corresponding
            dir indicated by ptr. Images are named according to img_ptr value 
        """
        if mode=="tr": 
            path=self.tr_path+str(self.tr_counter)+"/"
            counter = str(self.tr_counter)
        else: 
            path=self.ts_path+str(self.ts_counter)+"/"
        try:
            self.img_ptr += 1
            print "Path: ",path
            print "img_pointer: ",self.img_ptr
            cv.imwrite(path+str(self.img_ptr)+"_bw.png",samples[0])
            cv.imwrite(path+str(self.img_ptr)+"_depth.png",samples[1])
            cv.imwrite(path+str(self.img_ptr)+"_rgb.png",samples[2])
            np.save(path+str(self.img_ptr)+'_mtx.npy',samples[1])
            if mode=="tr": #Do not save index for testing as they have the same label
                self.index.save_sample(counter,name)
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

    def get_sample(self,mode,type,label,id):
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        if not type=="nmtx":
            bw_img = cv.imread(path+str(label)+"/"+str(id),0)
            return bw_img.reshape(bw_img.shape[0]*bw_img.shape[1])
        else:
            return np.load(path+str(label)+"/"+str(id)).reshape(192*256)

    def get_samples(self,mode="tr",type="bw"):
        #Mode selection
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        #
        #Type of file to be extracted
        if type == "bw":
            pattern = '*_bw.png'
        elif type=="mtx":
            pattern = '*_mtx.npy'
        elif type=="nmtx":
            pattern = '*_nmtx.npy'
        elif type=="rgb":
            pattern = '*_rgb.png'
        else:
            pattern = '*_depth.png'
        #
        samples_matrix = []; samples_labels = []
        """Building up the matrixes"""
        for label in os.listdir(path):
            #print "Label analyzed: ",label
            if not type=="rgb":
                for img in glob.glob1(path+str(label),pattern):
                    #print "File processed: ",img
                    if type=="nmtx" or type=="mtx":
                        img_vector = np.load(path+str(label)+"/"+str(img)).reshape(192*256)
                    else:
                        bw_img = cv.imread(path+str(label)+"/"+str(img),0)
                        img_vector = bw_img.reshape(bw_img.shape[0]*bw_img.shape[1])
                    try:
                        samples_matrix = np.vstack((samples_matrix,img_vector))
                        samples_labels = np.vstack((samples_labels,int(label)))
                    except:
                        samples_matrix = img_vector
                        samples_labels = int(label)    
            else:#When type is RGB
                for img in glob.glob1(path+str(label),pattern):
                    #print "File processed: ",img
                    samples_matrix.append(cv.imread(path+str(label)+"/"+str(img),0))
                    samples_labels.append(label)
        return samples_matrix,samples_labels
                    
    def save_samples(self,mode,type,data,labels):
        if mode == "tr": 
            path = self.tr_path
        else: 
            path = self.ts_path
        c_label = -1; counter = 1
        for i in xrange(data.shape[0]):
            if labels[i][0]!= c_label:
                c_label = labels[i][0];counter=1
            mtx = data[i].reshape(256,192)
            if type == "mtx":
                np.save(path+str(labels[i][0])+"/"+str(counter)+"_nmtx.npy",mtx)
                os.remove(path+str(labels[i][0])+"/"+str(counter)+"_mtx.npy") #Remove unnormalized matrix
            elif type == "bw":
                cv.imwrite(path+str(labels[i][0])+"/"+str(counter)+"_nbw.png",mtx)
                os.remove(path+str(labels[i][0])+"/"+str(counter)+"_bw.png") #Remove unnormalized bw img
            elif type == "depth":
                cv.imwrite(path+str(labels[i][0])+"/"+str(counter)+"_ndepth.png",mtx)
                os.remove(path+str(labels[i][0])+"/"+str(counter)+"_depth.png") #Remove unnormalized depth img
            counter+=1

    def save_sample(self,mode,type,label,id,img):
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        if not type=="nmtx":
            cv.imwrite(path+str(label)+"/"+str(id)+"_bw.png",img.reshape(192,256))
        else:
            np.save(path+str(label)+"/"+str(id)+"_nmtx.npy",img)

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

