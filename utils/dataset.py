import os, glob, cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

class ImageFolderFlat(Dataset):
    def __init__(self, root, img_size=224):
        self.paths = []
        for ext in ('*.png','*.jpg','*.jpeg','*.bmp'):
            self.paths += glob.glob(os.path.join(root, ext))
        self.T = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert('RGB')
        x = self.T(img)
        return x, os.path.basename(p)
