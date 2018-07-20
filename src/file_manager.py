"""
    File Manager module to take care of data dir
    Includes the following classes:
        - Index
        - Sample_Manager: to build the training and testing data sets.
        - Picture_Manager: to interact with the Recognizer Module.
        - External_DB_Manager: manages external DBs rather than the KI-FRS database
"""
import cv2 as cv
import os
from os.path import *
from os import listdir
import glob
import numpy as np
import datetime
from random import randint

class Index():
    """Index class to manage the index.txt file"""
    data_path = "%s/data/img/" % (os.path.abspath(os.path.join(os.getcwd(), os.pardir)))#Data path
    
    def __init__(self):
        """Always appends to the existing file. Creates a new one if it doesn't exist"""
        self.index_file = open(self.data_path+"index.txt","a+")
        self.index_file.close()
    
    def __del__(self):
        """Safely stores the text file when deleting the object"""
        try:
            self.index_file.close()
        except:
            pass
    
    def save_sample(self,id,name):
        """Given an ID number and the name, writes that data into the index file"""
        if self.get_sample_name(id)==None:
            self.index_file = open(self.data_path+"index.txt","a+")
            self.index_file.write(id+";"+name+";\n")
            self.index_file.close()
    
    def get_sample_name(self,id):
        """Given the label or ID, returns the name of the sample."""
        self.index_file = open(self.data_path+"index.txt","r")
        lines = self.index_file.readlines()
        self.index_file.close()
        for line in lines:
            f_id,name = line.split(";")[0],line.split(";")[1]
            if f_id==str(id):
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
    detect_path = "%s/detector/"%(data_path)
    #Images path
    img_path = "%s/img" % (data_path)
    #Training path
    tr_path = "%s/tr/" % (img_path)
    #Test path
    ts_path = "%s/ts/" %(img_path)
    #.xml file with cascade detector for frontal faces
    faceCascade = cv.CascadeClassifier(detect_path + "haarcascade_frontalface.xml")
    
    def __init__(self):
        """Initializes pointers to the next training or testing sample"""
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
        """Avoids creation of empty directories by deleting them"""
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
    #Detector path
    detect_path = "%sdetector/" % (data_path)
    #.xml file with cascade detector for depth images
    faceCascade = cv.CascadeClassifier(detect_path + "haarcascade_depth.xml")
    #Images path
    img_path = "%simg/" % (data_path)
    #Training path
    tr_path = "%str/" % (img_path)
    #Test path
    ts_path = "%sts/" % (img_path)
    #Recognition path
    rec_path = "%srecognition/"%(data_path)
    
    def get_sample_info(self,number):
        """Given a sample label or ID, returns the first black&white and depth images stored"""
        idx = Index()
        name = idx.get_sample_name(number)
        rgb_img = cv.imread(self.tr_path+str(number)+"/1_rgb.png")
        depth_img = cv.imread(self.tr_path+str(number)+"/1_depth.png")
        return name,rgb_img,depth_img
        
    def get_sample(self,mode,type,label,id):
        """Gets the specified sample given the complete definition of it"""
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        if not (type=="mtx" or type=="nmtx"):
            bw_img = cv.imread(path+str(label)+"/"+str(id),0)
            return bw_img.reshape(bw_img.shape[0]*bw_img.shape[1])
        else:
            #print "Needs to load a matrix"
            return np.load(path+str(label)+"/"+str(id)).reshape(192*256)

    def get_samples(self,mode="tr",type="bw"):
        """Generates a complete matrix either for training or test samples with the corresponding vector of labels"""
        #Mode selection
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        #Type of file to be extracted
        if type == "bw":
            pattern = '*_bw.png'
        elif type == "nbw":
            pattern = '*_nbw.png'
        elif type=="mtx":
            pattern = '*_mtx.npy'
        elif type=="nmtx":
            pattern = '*_nmtx.npy'
        elif type=="rgb":
            pattern = '*_rgb.png'
        elif type == "depth":
            pattern = '*_depth.png'
        else:
            pattern = '*_ndepth.png'
        samples_matrix = []; samples_labels = [];names=[]
        """Building up the matrixes"""
        for label in os.listdir(path):
            #print "Label analyzed: ",label
            if not type=="rgb":
                for img in glob.glob1(path+str(label),pattern):
                    names.append(img)
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
                    names.append(img)
                    samples_matrix.append(cv.imread(path+str(label)+"/"+str(img),cv.CV_LOAD_IMAGE_COLOR))
                    samples_labels.append(label)
        return samples_matrix,samples_labels,names
                    
    def save_samples(self,mode,type,data,labels,names=None):
        """Stores the preprocessed samples according to the type of file"""
        if mode == "tr": 
            path = self.tr_path
        else: 
            path = self.ts_path
        for i in xrange(data.shape[0]):
            if type == "mtx":
                mtx = data[i].reshape(256,192)
                np.save(path+str(labels[i])+"/"+names[i][0:(names[i].find("_"))]+"_nmtx.npy",mtx)
                #os.remove(path+str(labels[i][0])+"/"+str(counter)+"_mtx.npy") #Remove unnormalized matrix
            elif type == "bw":
                mtx = data[i].reshape(100,100)
                cv.imwrite(path+str(labels[i])+"/"+names[i][0:(names[i].find("_"))]+"_nbw.png",mtx)
                #os.remove(path+str(labels[i][0])+"/"+str(counter)+"_bw.png") #Remove unnormalized bw img
            elif type == "depth":
                mtx = data[i].reshape(100,100)
                cv.imwrite(path+str(labels[i])+"/"+names[i][0:(names[i].find("_"))]+"_ndepth.png",mtx)
                #os.remove(path+str(labels[i][0])+"/"+str(counter)+"_depth.png") #Remove unnormalized depth img

    def save_sample(self,mode,type,label,id,img):
        """Stores an individual file"""
        if mode=="tr":
            path = self.tr_path
        else:
            path = self.ts_path
        if not type=="nmtx":
            cv.imwrite(path+str(label)+"/"+str(id)+"_bw.png",img.reshape(192,256))
        else:
            np.save(path+str(label)+"/"+str(id)+"_nmtx.npy",img)

    def load_model(self,mode,source,num_components=None):
        """Loads a stored face recognition model. If it doesn't exists, returns None"""
        name = "eigenfaces_%s.yaml"%(source)
        model = None
        if mode==1 and num_components==None:    
            model = cv.createEigenFaceRecognizer()
        elif mode==1:
            model = cv.createFisherFaceRecognizer(num_components)
            return model
        if mode==2 and num_components==None:
            model = cv.createFisherFaceRecognizer()
        elif mode==2:
            model = cv.createFisherFaceRecognizer(num_components)
            return model
        try:
            model.load(self.rec_path+name)
            return model
        except:
            return None
        
    def save_model(self,mode,source,model):
        """Saves the given face recognition model"""
        if mode==1:
            name = "eigenfaces_%s.yaml"%(source)
        else:
            name = "fisherfaces_%s.yaml"%(source)
        model.save(self.rec_path+name)       

class External_DB_Manager():
    """Specialized class for managing description files for external databases"""
    data_path =  "%s/data/"%(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
    dbs_path = "%sextern_dbs/"%(data_path)
    att_db_path = "%satt/"%(dbs_path)
    mit_db_path = "%smit/"%(dbs_path)
    yale_db_path = "%syale/"%(dbs_path)
    
    def __init__(self,tr_percent):
        self.generate_indexes()
        self.build_sets(tr_percent)
    
    def generate_indexes(self):
        """Generates the index files for all the external databases"""
        
        """AT&T Database"""
        """Creates a file for each directory listing all the image files"""
        dirs = [x[0].split("/")[-1] for x in os.walk(self.att_db_path) if not x[0].split("/")[-1] == "."]
        set_file = open("%sset.dat"%(self.att_db_path),"w+")
        for d in dirs:
            if d == "":
                continue
            line = "%s%s/\n"%(self.att_db_path,d)
            set_file.writelines(line)
            tmp_file = open(line.rstrip("\n")+d+".info","w+")
            files = [x for x in os.listdir(line.rstrip("\n")) if x.endswith(".pgm")]
            for f in files:
                tmp_file.write(f+"\n")
            tmp_file.close()
        set_file.close()
       
        """YALE"""
        """Creates a file that lists all the available files for each individual"""
        dirs = [x[0].split("/")[-1] for x in os.walk(self.yale_db_path) if not x[0].split("/")[-1] == "."]
        set_file = open("%sset.dat"%(self.yale_db_path),"w+")
        for d in dirs:
            if d == "":
                continue
            line = "%s%s/\n"%(self.yale_db_path,d)
            set_file.writelines(line)
        set_file.close()
        
        """MIT"""
        """Creates the training and testing definition files as they are already divided"""
        tr_samples = sorted([x for x in os.listdir(self.mit_db_path+"tr/") if x.endswith(".pgm")])
        id = tr_samples[0][0:4]
        tmp_file = open(self.mit_db_path+"tr/"+id+".dat","w+")
        for sample in tr_samples:
            if not id==sample[0:4]:#New sample!
                tmp_file.close()
                id = sample[0:4]
                tmp_file = open(self.mit_db_path+"tr/"+id+".dat","w+")
            tmp_file.write(sample+"\n")
        tmp_file.close()
        ts_samples = sorted([x for x in os.listdir(self.mit_db_path+"ts/") if x.endswith(".pgm")])
        id = tr_samples[0][0:4]
        tmp_file = open(self.mit_db_path+"ts/"+id+".dat","w+")
        for sample in ts_samples:
            if not id==sample[0:4]:#New sample!
                tmp_file.close()
                id = sample[0:4]
                tmp_file = open(self.mit_db_path+"ts/"+id+".dat","w+")
            tmp_file.write(sample+"\n")
        tmp_file.close()

    def build_sets(self,tr_percent):
        """Creates training and testing files descriptors given the partition percentage of training images to preserve"""
        """AT&T & Yale"""
        paths = [self.att_db_path,self.yale_db_path]
        for path in paths:
            tr_file = open(path+"tr.dat","w+")
            ts_file = open(path+"ts.dat","w+")
            subjects = open(path+"set.dat","r").readlines()
            for sub in subjects:
                inf_file = "%s%s"%(sub.rstrip("\n"),[x for x in os.listdir(sub.rstrip("\n")) if x.endswith(".info")][0])
                samples = open(inf_file,"r").readlines()
                tr_file.write(sub.split("/")[-2]+"\n")
                ts_file.write(sub.split("/")[-2]+"\n")
                ts_num = len(samples)-int(len(samples)*(tr_percent/100.0))
                ts_lines =[]
                for i in xrange(ts_num):
                    rand = randint(0,len(samples)-1)
                    while ts_lines.count(rand)!=0:
                        rand = randint(0,len(samples)-1)
                    ts_lines.append(rand)
                    ts_file.write(sub.rstrip("\n")+samples[rand])
                    del samples[rand]
                for rest in samples:
                    tr_file.write(sub.rstrip("\n")+rest)
            tr_file.close()
            ts_file.close()    
        
        """MIT"""
        """No need to use the percentage, they are already divided"""
        tr_file = open(self.mit_db_path+"tr.dat","w+")
        ts_file = open(self.mit_db_path+"ts.dat","w+")
        dirs = ["tr","ts"]
        for dir in dirs:
            inf_files = [x for x in os.listdir(self.mit_db_path+dir+"/") if x.endswith(".dat")]
            for inf_file in inf_files:
                lines = open(self.mit_db_path+dir+"/"+inf_file,"r").readlines()
                if dir=="tr":
                    dest_file = tr_file
                else:
                    dest_file = ts_file
                dest_file.write(inf_file[0:4]+"\n")
                for line in lines:
                    dest_file.write(self.mit_db_path+dir+"/"+line)
        tr_file.close()
        ts_file.close()

    def get_data(self,database):
        """Given the database, reads the tr and ts description files and creates a list with the full path for each file and a vector for the labels"""
        tr_samples = []; ts_samples=[]
        tr_labels = []; ts_labels=[]
        if database=="att":
            tr_file = open(self.att_db_path+"tr.dat","r").readlines()
            ts_file = open(self.att_db_path+"ts.dat","r").readlines()
        elif database == "mit":
            tr_file = open(self.mit_db_path+"tr.dat","r").readlines()
            ts_file = open(self.mit_db_path+"ts.dat","r").readlines()
        else:
            tr_file = open(self.yale_db_path+"tr.dat","r").readlines()
            ts_file = open(self.yale_db_path+"ts.dat","r").readlines()
        label_counter = 1
        for line in tr_file:
            line = line.rstrip("\n")
            if not line.endswith(".pgm"):
                label_counter+=1
            else:
                tr_labels.append(label_counter)
                tr_samples.append(line)
        label_counter = 1
        for line in ts_file:
            line = line.rstrip("\n")
            if not line.endswith(".pgm"):
                label_counter+=1
            else:
                ts_labels.append(label_counter)
                ts_samples.append(line)
        return tr_samples,tr_labels,ts_samples,ts_labels
            
   
        
            
        