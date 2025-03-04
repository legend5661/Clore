from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import pandas as pd

class MyGlasDataset(ISDataset):
    def __init__(self, dataset_path, csv='train.csv', **kwargs):
        super(MyGlasDataset, self).__init__(**kwargs)
        
        self.dataset_path = Path(dataset_path) / csv
        
        dataset = pd.read_csv(self.dataset_path)
        
        self.dataset_samples = dataset['image']
        self.masks_paths = dataset['mask']

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self.dataset_path / image_name)
        
        mask_name = self.masks_paths[index]
        mask_path = str(self.dataset_path / mask_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        # print(f'image.shape:{image.shape}')
        # print(f'instances_mask.shape:{instances_mask.shape}')
        # instances_mask = (instances_mask - np.min(instances_mask)) / (np.max(instances_mask) - np.min(instances_mask))
        instances_mask[instances_mask > 0] = 1
        

        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)