import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import *
import argparse


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

          
args = parse_args()

DEFAULT_DIR = "Dataset_nomask/" +  args.month
Yield_path = "~/Documents/PhD/Donnees/dataframe/Corn_belt/GPP/GPPmt/yield"

class GLASS_TrainDataset(Dataset):
    def __init__(self,img_size):
        # self.img_dir = os.path.join(DEFAULT_DIR, "Train")
        self.img_dir = os.path.join(DEFAULT_DIR) #When we want to use all images in train
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
            image = np.where(np.isnan(image), -10, image)
            # Normalize each band separately
            min_pixel_values = np.nanmin(image, axis=(1, 2))
            max_pixel_values = np.nanmax(image, axis=(1, 2))
            image_normalized = (image.astype(np.float32) - min_pixel_values[:, np.newaxis, np.newaxis]) / (max_pixel_values[:, np.newaxis, np.newaxis] - min_pixel_values[:, np.newaxis, np.newaxis])
            # Standardize each band separately
        # image_tensor = torch.from_numpy(image_normalized)
        # mask = torch.isnan(image_tensor)
        # image_masked_tensor = masked_tensor(image_tensor, ~mask) 
        
        return image_normalized  
      
    def __len__(self):
        return self.nb_img


class GLASS_TestDataset(Dataset):
    def __init__(self,img_size):
        self.img_dir = os.path.join(DEFAULT_DIR)
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
            image = np.where(np.isnan(image), -10, image)
            # Normalize each band separately
            min_pixel_values = np.nanmin(image, axis=(1, 2))
            max_pixel_values = np.nanmax(image, axis=(1, 2))
            image_normalized = (image.astype(np.float32) - min_pixel_values[:, np.newaxis, np.newaxis]) / (max_pixel_values[:, np.newaxis, np.newaxis] - min_pixel_values[:, np.newaxis, np.newaxis])
            # Standardize each band separately
        # image_tensor = torch.from_numpy(image_normalized)
        # mask = torch.isnan(image_tensor)
        # image_masked_tensor = masked_tensor(image_tensor, ~mask)  

        return image_normalized  
    
    def __len__(self):
        return self.nb_img
 
 


def get_train_dataloader(args):
    if args.dataset == "GLASS":
        train_dataset = GLASS_TrainDataset(
            args.img_size
        )
    else:
        raise RuntimeError("No / Wrong dataset provided")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True)

    return train_dataloader, train_dataset


def get_test_dataloader(args):
    if args.dataset == "GLASS":
        test_dataset = GLASS_TestDataset(
            args.img_size
        )
    else:
        raise RuntimeError("No / Wrong dataset provided")

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle= False,pin_memory=True
        )

    return test_dataloader, test_dataset

# class Yield_Dataset(Dataset):
#     def __init__(self):
#         # Replace 'Yield_path' with the actual path to the directory containing the CSV file
#         self.data_dir = "Yield_path"
#         csv_path = os.path.join(self.data_dir, "yieldb_July_yearly_variation.csv")
#         self.data_frame = pd.read_csv(csv_path)

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, index):
#         # Retrieve the first column value (assuming column name is 'first_column_name')
#         Yield = torch.tensor(self.data_frame.iloc[index, 0], dtype=torch.float32)

        
#         return Yield
    

# # # Step 2: Create a custom data loader for LOOCV
# # class LeaveOneOutLoader:
# #     def __init__(self, dataset, batch_size=1):
# #         self.dataset = dataset
# #         self.batch_size = batch_size

# #     def __iter__(self):
# #         for i in range(len(self.dataset)):
# #             training_set = self.dataset[:i] + self.dataset[i + 1:]
# #             test_sample = self.dataset[i]

# #             training_loader = Yield_Dataset(training_set, batch_size=self.batch_size, shuffle=True)
# #             test_loader = Yield_Dataset([test_sample], batch_size=1)

# #             yield training_loader, test_loader

# #     def __len__(self):
# #         return len(self.dataset)
         

