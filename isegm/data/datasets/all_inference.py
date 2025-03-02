from pathlib import Path

import cv2

import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import pandas as pd


class AllforInferenceDataset(ISDataset):
    def __init__(self, dataset_path, csv='all_train.csv', dataset_name=None, **kwargs):
        super(AllforInferenceDataset, self).__init__(**kwargs)
        
        self.dataset_path = Path(dataset_path) / csv
        
        dataset = pd.read_csv(self.dataset_path)
        
        # self.dataset_samples = list(dataset['image'])
        # self.masks_paths = list(dataset['mask'])
        # self.dataset_samples = np.unique(self.dataset_samples)
        # self.masks_paths = np.unique(self.masks_paths)
        
        # self.dataset_samples = self.dataset_samples[46125:46175]
        # self.masks_paths = self.masks_paths[46125:46175]
        self.dataset_name = dataset_name
        self.dataset_samples = []
        self.masks_paths = []
        for img, msk in zip(dataset['image'], dataset['mask']):
            if dataset_name in img and dataset_name in msk:
                self.dataset_samples.append(img)
                self.masks_paths.append(msk)
        print(f'Length of image:{len(self.dataset_samples)}, mask:{len(self.masks_paths)}')


    def get_sample(self, index) -> DSample:
        lm = 10000  # 10000
        
        image_path = self.dataset_samples[index]
        
        mask_path = self.masks_paths[index]
        
        # if not image_path.startswith('staff'):
        #     image_path = '/staff/wanghn/torch_projects/ritm_interactive_segmentation/' + image_path
        #     mask_path = '/staff/wanghn/torch_projects/ritm_interactive_segmentation/' + mask_path
        
        # print(image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(f'mask path:{mask_path}')
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        
        width = image.shape[0]
        height = image.shape[1]
        while width >= lm or height >= lm:
            if width >= lm:
                width = width // 2
            if height >= lm:
                height = height // 2
                
        if width != image.shape[0] or height != image.shape[1]:
            image = cv2.resize(image, (width, height))
            instances_mask = cv2.imread(mask_path)
            instances_mask = cv2.resize(instances_mask, (width, height))
            instances_mask = instances_mask[:, :, 0].astype(np.int32)
        
        # print(f'image.shape:{image.shape}')
        # print(f'instances_mask.shape:{instances_mask.shape}')
        # instances_mask = (instances_mask - np.min(instances_mask)) / (np.max(instances_mask) - np.min(instances_mask))
        instances_mask[instances_mask > 0] = 1
        

        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)