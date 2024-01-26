import numpy as np
import re
import os
from base_operation import *

# 
class KittiDetectionDataset:
    """Creation of  detection Dataset """

    def __init__(self,root_path,label_path = None):
        """Initialization Function to read all paths

        Args:
            root_path : root path of dataset files
            label_path : path to label files 
        """
        self.root_path = root_path
        self.velo_path = os.path.join(self.root_path,"velodyne")
        self.image_path = os.path.join(self.root_path,"image_2")
        self.calib_path = os.path.join(self.root_path,"calib")
        if label_path is None:
            self.label_path = os.path.join(self.root_path, "label_2")
        else:
            self.label_path = label_path

        self.all_ids = os.listdir(self.velo_path)

    def __len__(self):
        """Function to return number of samples """
        
        return len(self.all_ids)
    
    # Get index function
    def __getitem__(self, item):
        """ Function to get items """

        name = str(item).zfill(6)

        velo_path = os.path.join(self.velo_path,name+'.bin')
        image_path = os.path.join(self.image_path, name+'.png')
        calib_path = os.path.join(self.calib_path, name+'.txt')
        label_path = os.path.join(self.label_path, name+".txt")

        P2,V2C = read_calib(calib_path)
        points = read_velodyne(velo_path,P2,V2C)
        image = read_image(image_path)
        labels,label_names = read_detection_label(label_path)
        labels[:,3:6] = cam_to_velo(labels[:,3:6],V2C)[:,:3]

        return P2,V2C,points,image,labels,label_names