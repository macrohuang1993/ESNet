from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from utils.preprocess import *
from torchvision import transforms
import time
from dataloader.EXRloader import load_exr

class DrivingStereoDataset(Dataset):

    def __init__(self, txt_file, root_dir, phase='train', load_disp=True, scale_size=(448, 896), same_LR_aug = False):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
            scale_size: Affect if phase=test/detect, used to scale spatially the input image to a scale divisible by network downsampling rate.
            self.img_size: Affect for phase=test/detect case, image size of the input image, used to scale back spatially the output disparity map (scaled by the scale_size parameter above) to the self.img_size. 
        """
        with open(txt_file, "r") as f:
            self.imgPairs = f.readlines()

        self.root_dir = root_dir
        self.phase = phase
        self.load_disp = load_disp
        self.scale_size = scale_size
        self.img_size = (400, 881)
        self.same_LR_aug = same_LR_aug
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size

    def __len__(self):
        return len(self.imgPairs)

    def __getitem__(self, idx):

        img_names = self.imgPairs[idx].rstrip().split()

        img_left_name = os.path.join(self.root_dir, img_names[0])
        img_right_name = os.path.join(self.root_dir, img_names[1])
        if self.load_disp:
            gt_disp_name = os.path.join(self.root_dir, img_names[2])

        def load_rgb(filename):

            img = None
            img = io.imread(filename)
            h, w, c = img.shape
            return img
           
        def load_disp(filename):
            gt_disp = None
            gt_disp = Image.open(gt_disp_name)
            gt_disp = np.ascontiguousarray(gt_disp,dtype=np.float32)/256

            return gt_disp

        s = time.time()
        img_left = load_rgb(img_left_name)
        img_right = load_rgb(img_right_name)
        if self.load_disp:
            gt_disp = load_disp(gt_disp_name)

        #print("load data in %f s." % (time.time() - s))

        s = time.time()
        if self.phase == 'detect' or self.phase == 'test':
            img_left = transform.resize(img_left, self.scale_size, preserve_range=True)
            img_right = transform.resize(img_right, self.scale_size, preserve_range=True)

            # change image pixel value type ot float32
            img_left = img_left.astype(np.float32)
            img_right = img_right.astype(np.float32)
            #scale = RandomRescale((1024, 1024))
            #sample = scale(sample)

        if self.phase == 'detect' or self.phase == 'test':
            rgb_transform = default_transform()
        else:
            rgb_transform = inception_color_preproccess()
            
        if self.same_LR_aug:
            H,W = img_left.shape[:2]
            #Concate along width to have same bright augmentation
            im = rgb_transform(np.concatenate((img_left,img_right),axis=1))
            img_left, img_right = im[:,:,:W], im[:,:,W:] 
        else:
            img_left = rgb_transform(img_left)
            img_right = rgb_transform(img_right)

        if self.load_disp:
            gt_disp = gt_disp[np.newaxis, :]
            gt_disp = torch.from_numpy(gt_disp.copy()).float()

        if self.phase == 'train':

            h, w = img_left.shape[1:3]
            #th, tw = 384, 768 
            th, tw = 256, 768 
            top = random.randint(120, h - th)
            left = random.randint(0, w - tw)

            img_left = img_left[:, top: top + th, left: left + tw]
            img_right = img_right[:, top: top + th, left: left + tw]
            if self.load_disp:
                gt_disp = gt_disp[:, top: top + th, left: left + tw]



        sample = {  'img_left': img_left, 
                    'img_right': img_right, 
                    'img_names': img_names
                 }

        if self.load_disp:
            sample['gt_disp'] = gt_disp

        #print("deal data in %f s." % (time.time() - s))

        return sample

