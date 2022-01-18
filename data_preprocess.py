from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class Preprocess_Dataset(Dataset):
    def __init__(self, dir_B, dir_A, convertor=None):
        self.dir_A, self.dir_B = dir_A, dir_B
        self.convertor = transforms.Compose(convertor)

        self.B_images = os.listdir(self.dir_B)
        self.A_images = os.listdir(self.dir_A)
        self.B_len = len(self.B_images)
        self.A_len = len(self.A_images)

    def __getitem__(self, index):
        A_image = np.array(Image.open(os.path.join(self.dir_A, self.A_images[index % self.A_len])).convert("RGB"))
        B_image = np.array(Image.open(os.path.join(self.dir_B, self.B_images[index % self.B_len])).convert("RGB"))

        item_A = self.convertor(A_image)
        item_B = self.convertor(B_image)

        return (item_B, item_A)

    def __len__(self):
        return max(len(self.B_images), len(self.A_images))