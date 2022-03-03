import numpy as np
import pandas as pd
import os
import cv2
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class image_loader():
    
#supported extension,you can add more extension
    extensions: tuple =\
        [".jpg"]
    def __init__(self,path:str):
        self.path=path
        
        self.data=self.parse_dir(self.path)
        
#returns the path of all the labels in the folder
    def parse_dir(self,path):
        
        if os.path.isfile(self.path):
            assert path.lower.endswith(self.extensions
                                      ),f"only type {self.extensions}, are supported"
            return self.path
        
        if os.path.isdir(self.path):
            paths = [os.path.join(path,img) for img in os.listdir(path)]
            return paths
        
#returns the path of each image with its label
    def parse_dir_with_label(self,label_df):
        if "id" and "has_cactus" not in label_df.columns.values:
            raise Exception(f"id or label not found in {label_df.columns},please rename image name column to id and image label to label ")
        
        data_with_label=[]
        index=label_df.index

        for path in self.data:
            
            cond=label_df['id']==os.path.basename(path)
            idx=index[cond]
            label=label_df.iloc[idx].values[0,-1]
            
            data_with_label.append((path,label))  
            
        
            
        return data_with_label
    
# the 'process_image_with_label' function is very slow on large datasets
# and its purpose is only for the kaggle problem given,
# please if you have any faster alternatives feel free to message me @ bmtukur621@gmail.com

# it automatically read image and store it with its label

#e.g data=pd.readcsv('data_dir')
#    img_ldr=image_loader(path)
#    img,label=img_ldr.process_image_with_label(data)
#

    def process_image_with_label(self,label_df):
        img_with_labels=self.parse_dir_with_label(label_df)
        
        images=[]
        
        if len(img_with_labels) <1:
            raise Exception("no image processed, please recheck the images")
            
        for img_path,label in img_with_labels:
            img=cv2.imread(img_path)
            if img is not None:
                images.append((img,label))
        
        return images
    
#this functions reads the image and return the image with its id (name)
#it's relatively fast                    

    def process_image_no_label(self):
        
        images=[]
        
        if len(self.data)<1:
            raise Exception("no image found in folder")
        for img_path in self.data:
            img=cv2.imread(img_path)
            if img is not None:
                images.append((img,os.path.basename(img_path)))
                
        return images


