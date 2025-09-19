# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import os.path
import os.path
import torchvision.transforms as transforms
from PIL import Image
import PIL
import random
import numpy as np
import scipy.io as sio
import torch

import hdf5storage
from os import listdir
from os.path import join

from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import Compose,  RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomApply, Resize

from pathlib import Path

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def target_transform_crop(configs):
    return Compose([   
        Resize([configs.data.image_size, configs.data.image_size], Image.BICUBIC),
        RandomCrop(configs.data.crop_size),      
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
   #    RandomApply(random_rotation,p=0.5)
    ])


def target_transform_resize_crop(configs):
    return Compose([   
        Resize([configs.data.image_size, configs.data.image_size], Image.BICUBIC),
        #RandomCrop(configs.data.crop_size),      
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
   #    RandomApply(random_rotation,p=0.5)
    ])


def target_transform_resize(configs):
    return Compose([   
        Resize([configs.data.image_size, configs.data.image_size], Image.BICUBIC),
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def load_mat(filepath, normal_Option):
    #img = loadmat(filepath) # path of .mat files to be loaded
    img = hdf5storage.loadmat(filepath)
    matname = list(img.keys())[0] # take variable name (defined in Matlab)
    img = np.array(img[matname])
    img = img[:,:,np.newaxis]
    img = img.astype("float32")
    ###################### do normalization here ##############################
    
    if normal_Option:
        #standard
        # std = np.std(img)
        # mean = np.mean(img)
        # img = (img-mean)/std    
        
        #normalization
        min = np.min(img)
        max = np.max(img)
        img = (img-min)/(max-min)    
    
  #  img = (img+1000)/(7000)
     
    ###################### do normalization here ##############################    
    ToPIL = transforms.ToPILImage()
#    mode = 'F'    
    img = ToPIL(img)
    return img

#수정


class Dental_LD(Dataset):
    def __init__(self, root, configs):
        self.root = configs.data.root
        self.dir = os.path.join(root, 'target')

        #self.transform = target_transform()
        if configs.data.resize_or_crop == 'resize_and_crop':
            self.target_transform = target_transform_resize_crop(configs)
        else:
            self.target_transform = target_transform_crop(configs)

        self.crop_size = configs.data.crop_size
        self.normal_Option = configs.data.normal_Option
        self.remove_back_patch = configs.data.remove_back_patch
               
        self.image_filenames = [join(self.dir, x) for x in listdir(self.dir) if is_image_file(x)] # load files from filelist.
   
        
    def __getitem__(self, index):
        if self.remove_back_patch:
            count = 0
            while count <1:
                seed = random.randint(0,2**16)
                
                random.seed(seed)        
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                in_image = self.target_transform(load_mat(self.image_filenames[index], self.normal_Option))

                if np.logical_or((sum(sum(np.array(in_image)<-800)) > (self.crop_size**2)/2.5 ),(sum(sum(np.array(in_image)<-800)) > (self.crop_size**2)/2.5 )) :
                    continue
                else: 
                    count += 1  

            t = transforms.ToTensor()
            target = t(in_image)
            
            return target
        else:
            
            seed = random.randint(0,2**16)
                
            random.seed(seed)        
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            in_image = self.target_transform(load_mat(self.image_filenames[index], self.normal_Option))
                
            t = transforms.ToTensor()
            target = t(in_image)

            return target

    def __len__(self):
        return len(self.image_filenames)
    
    def name(self):
        return 'DentalDataset'
    

class Dental_LD_TEST(Dataset):
    def __init__(self, root, configs):
        self.root = configs.data.root
        self.dir_in = os.path.join(root, 'in', configs.data.test_in)
        self.dir_target = os.path.join(root, 'target', configs.data.test_target)    


        self.normal_Option = configs.data.normal_Option 
        self.image_in_filenames = [join(self.dir_in, x) for x in listdir(self.dir_in) if is_image_file(x)] # load files from filelist.
        self.image_target_filenames = [join(self.dir_target, x) for x in listdir(self.dir_target) if is_image_file(x)] # load files from filelist.
        self.target_transform = target_transform_resize(configs)
        
        
    def __getitem__(self, index):
       
        in_image = self.target_transform(load_mat(self.image_in_filenames[index], self.normal_Option))       
        target_image = self.target_transform(load_mat(self.image_target_filenames[index], self.normal_Option))    

        t = transforms.ToTensor()
        return t(in_image), t(target_image)

    def __len__(self):
        return len(self.image_in_filenames)
    def name(self):
        return 'DentalDataset_TEST'    


def create_dataloader(configs, evaluation=False):
  shuffle = True if not evaluation else False
  if configs.data.dataset == 'Dental_LD' and evaluation== False:
  
    train_dataset = Dental_LD(Path(configs.data.root) / f'Training', configs)
    val_dataset = Dental_LD(Path(configs.data.root) / f'Validation', configs)
    train_loader = DataLoader(
     dataset=train_dataset,
     batch_size=configs.training.batch_size,
     shuffle=shuffle,
     drop_last=True
    )
    val_loader = DataLoader(
     dataset=val_dataset,
     batch_size=configs.training.batch_size,
     # shuffle=False,
     shuffle=True,
     drop_last=True
    )
    return train_loader, val_loader
  
  elif configs.data.dataset == 'Dental_LD' and evaluation== True:  

    test_dataset = Dental_LD_TEST(Path(configs.data.root) / f'Test'/ f'1)Same', configs)
    test_loader = DataLoader(
     dataset=test_dataset,
     batch_size=configs.eval.batch_size,
     shuffle=shuffle,
     drop_last=False
    )
    return test_loader
  else:
    raise ValueError(f'Dataset {configs.data.dataset} not recognized.')
  


  #return train_loader, val_loader

