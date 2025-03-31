from typing import Dict, Callable, Tuple

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


class PatchDataset(Dataset):
    """ Dataset from a WSI patcher to directly read tiles on a slide  """

    def __init__(self, patch_dir, transforms: Dict[str, Callable] = None):
        self.transforms = transforms
        self.patches = list(patch_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        transformed_imgs = dict()
        filename = self.patches[index]
        x, y = filename.name.split(".")[0].split("_")
        x = int(x)
        y = int(y)
        coords = np.array([x, y])
        img = Image.open(filename)

        for name, transform in self.transforms.items():
            img_t = transform(img)
            transformed_imgs[name] = img_t

        return transformed_imgs, coords
