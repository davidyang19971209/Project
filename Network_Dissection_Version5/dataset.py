import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from utils import crop_sample, pad_sample, resize_sample, normalize_volume, load_nii, make_one_hot, convert_mask_to_one
from IPython import embed


class BrainSegmentationDataset(Dataset):
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 4
    out_channels = 4

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=0,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        #dirpath 是当前目录， dirnames，是目录下的文件夹，filenames， 是目录下的文件
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            mask_path = ""
            #filter 来筛选名字带.tif的文件
            #key指按照某一项排序
            #filter 来筛选名字带.tif的文件
            #key指按照某一项排序
            for filename in sorted(filter(lambda f: ".gz" in f, filenames)):  
                if "seg" in filename:
                    mask_path = os.path.join(dirpath,filename)
                    mask_slices.append(load_nii(mask_path))
                else:
                    filepath = os.path.join(dirpath, filename) 
                    image_slices.append(load_nii(filepath))

            embed()
            #只筛选带有肿瘤的slice
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]

                volumes[patient_id] = np.array(image_slices).transpose(1,2,3,0)
                masks[patient_id] = np.array(mask_slices).transpose(1,2,3,0)

            embed()

        #patient 是一个字典，里面是patient_id和其对应的image(无mask)
        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            #分出validation set
            validation_patients = random.sample(self.patients, k=validation_cases)                                      #注意K有可能超
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]
        embed()

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [(s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights]
        
        print("one hotting {} masks...".format(subset))
        self.volumes = [(v, make_one_hot(m)) for v,  m in self.volumes]
        embed()

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]
        embed()

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v,  m in self.volumes]
        embed()

        print("one hotting {} masks...".format(subset))
        self.volumes = [(v, convert_mask_to_one(m)) for v,  m in self.volumes]
        embed()

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling
        self.transform = transform
        embed()

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor
