from pathlib import Path

import cv2

import numpy as np

from isegm.data.base_copy import ISDataset
from isegm.data.sample import DSample
import pandas as pd


class AllDataset(ISDataset):
    def __init__(self, dataset_path, csv='all_train.csv', **kwargs):
        super(AllDataset, self).__init__(**kwargs)
        
        self.dataset_path = Path(dataset_path) / csv
        
        dataset = pd.read_csv(self.dataset_path)
        
        self.dataset_samples = dataset['image']
        self.masks_paths = dataset['mask']

    def get_sample(self, index) -> DSample:
        image_path = self.dataset_samples[index]
        
        mask_path = self.masks_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(f'mask path:{mask_path}')
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        # print(f'image.shape:{image.shape}')
        # print(f'instances_mask.shape:{instances_mask.shape}')
        # instances_mask = (instances_mask - np.min(instances_mask)) / (np.max(instances_mask) - np.min(instances_mask))
        instances_mask[instances_mask > 0] = 1
        

        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)