import config

import torch 
import torch.nn as nn 
import numpy as np


class dataloader(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = data[1]
        image = np.array(data[0])
        if self.transforms:
            image = self.transforms(image=image)['image']

        image = torch.tensor(image, dtype=torch.float)
        image = image.unfold(0, config.patch_size, config.patch_size).unfold(1, config.patch_size, config.patch_size)
        image = image.reshape(image.shape[0], image.shape[1], image.shape[2]*image.shape[3]*image.shape[4])
        image = image.view(-1, image.shape[-1])

        return {
            'patches' : image,
            'label' : torch.tensor(label, dtype=torch.long),
        }