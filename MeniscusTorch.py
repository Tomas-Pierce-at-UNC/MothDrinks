from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
import json
import pandas as pd
from skimage.io import imread
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim

J_FILENAME = "/home/tomas/Projects/DrinkMoth2/cocoMen/CocoMeniscus/annotations/instances_default.json"
IMG_DIR = "/home/tomas/Projects/DrinkMoth2/cocoMen/CocoMeniscus/images/"


class MeniscusPositionDataset(Dataset):

    def __init__(self, annotations_file=J_FILENAME, img_dir=IMG_DIR, transform=None, target_transform=None):
        with open(annotations_file) as annote:
            labeldata = json.load(annote)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self._licenses = labeldata['licenses']
        self._info = labeldata['info']
        self._categories = labeldata['categories']
        self._images = labeldata['images']
        self._annotations = labeldata['annotations']

        self.categories = pd.DataFrame(self._categories)
        self.images = pd.DataFrame(self._images)
        self.annotations = pd.DataFrame(self._annotations)
        del self.annotations['segmentation']
        del self.annotations['attributes']
        del self.annotations['iscrowd']
        boxes = np.array(list(map(np.array, self.annotations.bbox)))
        self.annotations['bbox0'] = boxes[:,0]
        self.annotations['bbox1'] = boxes[:,1]
        self.annotations['bbox2'] = boxes[:,2]
        self.annotations['bbox3'] = boxes[:,3]
        del self.annotations['bbox']
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        img_index = annotation.image_id
        img_row = self.images[self.images.id == img_index].iloc[0]
        img_name = img_row.file_name
        abs_filename = self.img_dir + img_name
        img_arr = imread(abs_filename)
        img = torch.from_numpy(img_arr)
        annotation = torch.from_numpy(annotation.values)
        if self.transform:
            img = self.transform(img_arr)
        if self.target_transform:
            label = self.target_transform(annotation)
        return img, annotation

class MeniscusNetwork(nn.Module):

    def __init__(self):
        super(MeniscusNetwork, self).__init__()
        self.linear = nn.Linear(28*28, 10)
    def forward(self, x):
        out = self.linear(x)
        return out

training_data = MeniscusPositionDataset(transform=ToTensor())
train_dataloader = DataLoader(training_data, batch_size=20, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = MeniscusNeuralNetwork().to(device)
