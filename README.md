# WSI_SR_CL

## 😎 Introduction
This repository is the official implementation of our 

**JOINT SUPER-RESOLUTION AND TISSUE PATCH
CLASSIFICATION FOR WHOLE SLIDE HISTOLOGICAL IMAGES** 

[[paper link]()]

*[Zh. Sun*, A. Khvostikov, A. Krylov, A. Sethi, I. Mikhailov,P. Malkov]

> Segmentation of wholeslide histological images through the classification of tissue types of small fragments is an extremely relevant task in digital pathology, necessary for the development of methods for automatic analysis of wholeslide histological images. The extremely large resolution of such images also makes the task of increasing image resolution relevant, which allows storing images at a reduced resolution and increasing it if necessary. Annotating whole slide images by histologists is complex and time-consuming, so it is important to make the most efficient use of the available data, both labeled and unlabeled.
In this paper we propose a novel neural network method to simultaneously solve the problems of super-resolution of histological images from 20x optical magnification to 40x and classifying image fragments into tissue types at 20x magnification. The use of a single encoder as well as the proposed neural network training scheme allows to achieve better results on both tasks compared to existing approaches. The PATH-DT-MSU WSS2v2 dataset presented for the first time in this paper was used for training and testing the method. On the test sample, an accuracy value of 0.971 and a balanced accuracy value of 0.916 were achieved in the classification task on 5 tissue types, for the super-resolution task, values of PSNR=32.26 and SSIM=0.89 were achieved. 

## Prepare the environment
Use wheel to download the [psimage](https://github.com/xubiker/psimage) package.

and then 
```pip install -r requirements.txt ```

## Prepare the data 
This project uses a dataset of whole slide histological images [PATH-DT-MSU WSS2v2](https://imaging.cs.msu.ru/en/research/histology/path-dt-msu) with polygonal annotation of sections by tissue type. This set of images was collected and prepared by the Laboratory of Mathematical Methods of Image Processing, Faculty of Computational Mathematics and Cybernetics, Lomonosov Moscow State University, and the Department of Pathology, Medical Research and Education Center, Lomonosov Moscow State University.

You need download the [WSS2v2 set](https://disk.yandex.ru/d/Z8juX1qDbnq88w) and [annation](https://disk.yandex.ru/d/r6GFEV2QI47MGQ) into your disk and take its as follow structure:
```
WSS2_v2_psi
├── test
│   ├── test_01.psi
│   ├── test_01_v2.5.json
│   ├── test_02.psi
│   ├── test_02_v2.5.json
│   ├── test_03.psi
│   ├── test_03_v2.5.json
│   ├── test_04.psi
│   ├── test_04_v2.5.json
│   ├── test_05.psi
│   └── test_05_v2.5.json
└── train
    ├── train_01.psi
    ├── train_01_v2.5.json
    ├── train_02.psi
    ├── train_02_v2.5.json
    ├── train_03.psi
    ├── train_03_v2.5.json
    ├── train_04.psi
    ├── train_04_v2.5.json
    ├── train_05.psi
    └── train_05_v2.5.json
 
```

After that, you need to run ```split_dataset.py```(you need change ```ds_path``` in the file) to split the test data in order to better test the model.

## Train the model 
### First step: pretrain the gan model: 
Download the pretrained ckpt in [real-resgan project]{https://github.com/xinntao/Real-ESRGAN/tree/master}

```
python main_dual.py \
--batch_size 16 \
--device 2 \
--nb_classes 6 \
--drop_out 0.2 \
--blr 1e-5 \
--epochs 100 \
--train_gan \
--train_data_path WSS2_v2_psi/train/ \
--val_data_path data_v2.5 \
--discriminator_path RealESRGAN_x2plus_netD.pth \
--pretrain_rrdb RealESRGAN_x2plus.pth \
--runs_name gan
```

### Second step: fine-tune on cls and sr task
```
python main_dual.py --batch_size 32 \
--device 3 \
--nb_classes 5 \
--pretrain_rrdb output_dir/gan/checkpoint-99.pth \
--train_data_path WSS2_v2_psi/train/ \
--val_data_path data_v2.5 \
--drop_out 0.5 \
--blr 1e-4 \
--epochs 50 \
--smoothing 0.1 \
--balance_coeff 0.3\
--runs_name v2.5_train_dual_20x
```

