import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch 

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob(os.path.join(root, mode) + '/*.*'))
        if mode == 'train':
            self.files.extend(sorted(glob.glob(os.path.join(root, 'test') + '/*.*')))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w/2, h))
        img_B = img.crop((w/2, 0, w, h))

        #if np.random.random() < 0.5:
        if img.mode == 'I;16':
            img_A = np.array(img_A)[:, :]
            img_B = np.array(img_B)[:, :]
            #img_A = Image.fromarray((img_A - img_A.min())/(img_A.max()-img_A.min()),'L')
            #img_B = Image.fromarray((img_B - img_B.min())/(img_B.max()-img_B.min()),'L')
            img_A = 2*(img_A - img_A.min())/(img_A.max()-img_A.min())-1
            img_B = 2*(img_B - img_B.min())/(img_B.max()-img_B.min())-1
            #img_A = torch.from_numpy(img_A)
            #img_B = torch.from_numpy(img_B)
            img_A = torch.DoubleTensor(img_A).reshape((1,w/2,h))
            img_B = torch.DoubleTensor(img_B).reshape((1,w/2,h))
            img_A = img_A[:,:,:]
            img_B = img_B[:,:,:]
            #img_A = Image.fromarray(np.array(img_A)[::-1, ::-1], 'I;16')
            #img_B = Image.fromarray(np.array(img_B)[::-1, ::-1], 'I;16')
            #img_A = img_A.convert('RGB')
            #img_B = img_B.convert('RGB')
        elif img.mode == 'L':
            img_A = Image.fromarray(np.array(img_A)[:,:], 'L')
            img_B = Image.fromarray(np.array(img_B)[:,:], 'L')
        else:
            img_A = img_A.convert('L')
            img_B = img_B.convert('L')
            #img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            #img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')
        if img.mode != 'I;16':
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return len(self.files)
