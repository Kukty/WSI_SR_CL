import os
from typing import Any
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torchvision
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np
import cv2
from datasets.psi_torch_dataset import TorchPSIDataset
from datasets.psi_dataset_regions_cls import (
    PSIRegionClsDataset,
    PSIRegionClsDatasetParams,
)
from pathlib import Path
def build_transform(is_train, args):
    # mean = IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda x: (
                    torch.from_numpy(x[0]).permute(2, 0, 1),
                    (torch.where(torch.from_numpy(x[1]) == 1 )[0]).squeeze()
                    
                    # (torch.where(torch.from_numpy(x[1]) == 1 )[0]).squeeze(),
                    # torch.from_numpy(x[1])
                )
            ),
            transforms.Lambda(
                lambda x:(

                    torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomRotation(45),
                    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    torchvision.transforms.RandomCrop(args.input_size),
                    ])(x[0]),
                    x[1]
                        )
            )

        ]
    )
        # # this should always dispatch to transforms_imagenet_train
        # transform = torchvision.transforms.Compose([
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandomVerticalFlip(),
        #         torchvision.transforms.RandomRotation(45),
        #         torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #         torchvision.transforms.RandomCrop(args.input_size),
        #         torchvision.transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        return transform

    transform = torchvision.transforms.Compose([

                torchvision.transforms.RandomCrop(args.input_size),
                torchvision.transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    return transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # root = args.train_data_path if is_train else args.val_data_path
    if is_train:
        if args.input_size==224:
            layer = 2
        elif args.input_size==112:
            layer = 4
        elif args.input_size==448:
            layer = 1 
        dataset = TorchPSIDataset(
        ds_constructor=PSIRegionClsDataset,
        ds_params=PSIRegionClsDatasetParams(
            path=Path("/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi/train/"),
            patch_size=args.input_size,
            layer=layer,
            region_intersection=0.7,
            balance_coeff=args.balance_coeff,
            # annotations_path=Path("/home/z.sun/wsi_SR_CL/annotation/WSS2_train/")
        ),
        n_procs=args.n_procs,
        queue_size=2000,
        transform=transform,
    )
    else :
        dataset = datasets.ImageFolder(args.val_data_path, transform=transform)
    # if "NCT-CRC-HE-100K" in args.train_data_path or "CRC-VAL-HE-7K" in args.train_data_path:
    #     train_size = int(0.8 * len(dataset))
    #     val_size = len(dataset) - train_size

    #     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    #     if is_train:
    #         return train_dataset
    #     else:
    #         return val_dataset

    # print(dataset)

    return dataset

class SR_CLA_Dataset(Dataset):
    def __init__(self,lr_data_path,hr_data_path,is_train=True,args=None) -> None:
        super().__init__()
        self.lr_dataset = datasets.ImageFolder(lr_data_path, transform=None)
        self.hr_dataset = datasets.ImageFolder(hr_data_path,transform=None)
        if is_train:
            self.transform = A.Compose(
                transforms=[
                    # A.RandomResizedCrop(args.input_size, args.input_size),
                    A.HorizontalFlip(),
                    A.Rotate(limit=30),
                    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    A.Normalize(mean=[0., 0., 0.], std=[1, 1, 1]),
                    ToTensorV2()
                ],
                additional_targets={'image1':'image'},
                is_check_shapes=False
            )
        else:
            self.transform = A.Compose(
                transforms=[
                    # A.RandomResizedCrop(args.input_size, args.input_size),
                    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    A.Normalize(mean=[0., 0., 0.], std=[1, 1, 1]),
                    ToTensorV2()
                ],
                additional_targets={'image1':'image'},is_check_shapes=False
            )
        
    def __len__(self):
        return self.hr_dataset.__len__()
    def __getitem__(self, i):
        hr_img,label_hr = self.hr_dataset.__getitem__(i)
        lr_img,label_lr = self.lr_dataset.__getitem__(i)
        assert label_hr == label_lr
        transformed = self.transform(image = np.array(hr_img),image1=np.array(lr_img))
        hr_img = transformed['image']
        lr_img = transformed['image1']
        return lr_img,hr_img,label_hr
    


# def load_image(img_path,if_normol=False):
#     image = cv2.imread(img_path)
#     imgae = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     image = image / 255.0
#     if if_normol:
#         image = (image - np.mean(image,axis=(0, 1))) / np.std(image,axis=(0, 1))

#     return image


# class WSIDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, mode='train'):
#         self.root_dir = root_dir
#         self.mode = mode
#         self.classes = ['AT', 'BG', 'LP', 'MM', 'TUM']
#         self.labels = []
#         self.jpg_paths = []

#         for dirpath, dirnames, filenames in os.walk(root_dir):
#             for filename in filenames:
#                 if filename.endswith('.jpg'):
#                     jpg_path = os.path.join(dirpath, filename)
#                     self.jpg_paths.append(jpg_path)
#         print(f"Successfully Load {self.jpg_paths.__len__()} images")
#         if self.mode == 'train':
#             # Add code for complex image augmentation here
#             self.transforms = torchvision.transforms.Compose([
#                 torchvision.transforms.RandomHorizontalFlip(),
#                 torchvision.transforms.RandomVerticalFlip(),
#                 torchvision.transforms.RandomRotation(45),
#                 torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#                 torchvision.transforms.RandomCrop(224),
#                 torchvision.transforms.ToTensor()
#             ])
#         elif self.mode == 'val':
#             self.transforms = torchvision.transforms.Compose([ torchvision.transforms.ToTensor()
#             ])

#         else:
#             raise ValueError("Invalid mode. Mode must be 'train' or 'val'.")

#     def __len__(self):
#         return len(self.jpg_paths)

#     def __getitem__(self, index):
#         image = load_image(self.jpg_paths[index])

#         label = self.jpg_paths[index].split('/')[-2]
#         self.classes = ['AT', 'BG', 'LP', 'MM', 'TUM']
#         label = self.classes.index(label)
#         image = self.transforms(image)

#         return image, label

# import PIL.Image

# def __getitem__(self, index):
#     image = load_image(self.jpg_paths[index])
#     image = PIL.Image.fromarray(image)  # Convert numpy array to PIL Image

#     label = self.jpg_paths[index].split('/')[-2]
#     self.classes = ['AT', 'BG', 'LP', 'MM', 'TUM']
#     label = self.classes.index(label)

#     image = self.transforms(image)

#     return image, label