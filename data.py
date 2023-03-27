import os
import sys
import torch
import numpy as np
from PIL import Image
from os import listdir
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from utils.tools import default_loader, is_image_file, normalize


class BinaryToTensor:
    """Convert a binary PIL Image to a tensor.

    Converts a binary PIL Image (H x W) to a FloatTensor of shape (1 x H x W).
    """
    
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        
        if img.mode != '1':
            raise ValueError('img should be binary (mode "1"). Got mode "{}"'.format(img.mode))

        # Convert PIL Image to PyTorch tensor
        img = F.to_tensor(img)

        # Add extra dimension for batch size (1)
        # img = img.unsqueeze(0)

        return img

    def _is_tensor_image(self, img):
        return isinstance(img, torch.Tensor)



class Dataset(data.Dataset):
    def __init__(self, data_path, mask_path):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.mask_path = mask_path
        self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.masks = [x for x in listdir(mask_path) if is_image_file(x)]


    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = default_loader(path)
        mask_p = os.path.join(self.mask_path, self.masks[index])
        mask = default_loader(mask_p, False)
        # print(f"/////// {np.unique(np.array(mask))} ////////")
        # transformations
        img = transforms.Resize((512, 512))(img)
        mask = transforms.Resize((512, 512))(mask)
        
        img = transforms.ToTensor()(img)  # turn the image to a tensor
        img = normalize(img)
        mask =BinaryToTensor()(mask)
        # mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        # print(f"/////// {np.unique(np.array(mask))} ////////")
        # print("**************************")
        # mask = normalize(mask)
        return img, mask

    def __len__(self):
        return len(self.samples)