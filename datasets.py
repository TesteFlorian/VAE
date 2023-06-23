import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import glob
#from torch.masked import masked_tensor, as_masked_tensor
import warnings

# Disable prototype warnings and such
warnings.filterwarnings(action='ignore', category=UserWarning)

import os
import rasterio
import numpy as np
import torch


DEFAULT_DIR = "Dataset/July"

class GLASS_TrainDataset(Dataset):
    def __init__(self,img_size):
        self.img_dir = os.path.join(DEFAULT_DIR, "Train")
        self.img_files = [os.path.join(self.img_dir, img)
                          for img in os.listdir(self.img_dir)
                          if (os.path.isfile(os.path.join(self.img_dir, img)) and img.endswith('tif'))]
        self.nb_img = len(self.img_files)
    
    def __getitem__(self, index):
        image_path = self.img_files[index]
        with rasterio.open(image_path) as src:
            # Read the image as a multi-dimensional NumPy array
            image = src.read()
            # Replace NaN values with 0, images values start from 0
            image = np.where(np.isnan(image), 0, image)
            # Normalize each band separately
            max_pixel_values = np.nanmax(image, axis=(1, 2))
            image_normalized = (image.astype(np.float32) / max_pixel_values[:, np.newaxis, np.newaxis]).astype(np.float32)
        # image_tensor = torch.from_numpy(image_normalized)
        # mask = torch.isnan(image_tensor)
        # image_masked_tensor = masked_tensor(image_tensor, ~mask) 
        
        return image_normalized  
      
    def __len__(self):
        return self.nb_img


class GLASS_TestDataset(Dataset):
    def __init__(self,img_size):
        self.img_dir = os.path.join(DEFAULT_DIR, "Test")
        self.img_files = [os.path.join(self.img_dir, img)
                          for img in os.listdir(self.img_dir)
                          if (os.path.isfile(os.path.join(self.img_dir, img)) and img.endswith('tif'))]
        self.nb_img = len(self.img_files)
    
    def __getitem__(self, index):
        image_path = self.img_files[index]
        with rasterio.open(image_path) as src:
            # Read the image as a multi-dimensional NumPy array
            image = src.read()
            # Replace NaN values with 0, images values start from 0
            image = np.where(np.isnan(image), 0, image)
            # Normalize each band separately
            max_pixel_values = np.nanmax(image, axis=(1, 2))
            image_normalized = (image.astype(np.float32) / max_pixel_values[:, np.newaxis, np.newaxis]).astype(np.float32)
        # image_tensor = torch.from_numpy(image_normalized)
        # mask = torch.isnan(image_tensor)
        # image_masked_tensor = masked_tensor(image_tensor, ~mask)  

        return image_normalized  
    
    def __len__(self):
        return self.nb_img
