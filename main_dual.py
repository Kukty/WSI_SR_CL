import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import random
from basicsr.losses.basic_loss import  PerceptualLoss
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from torchvision import models,transforms
from datasets.utils import low_entropy_filter
from timm.models.layers import trunc_normal_
# Use Later
import albumentations as A 
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from datasets.psi_dataset_random_sr_cls import (
    PSIRegionSrClsDatasetParams,
    PSIRegionSrClsDataset,
)

from datasets.psi_torch_dataset import TorchPSIDataset
from wsiDataset import build_dataset,SR_CLA_Dataset
import lr_decay as lrd
import misc
from models_esr import models_esrc
from misc import NativeScalerWithGradNormCount as NativeScaler
from engine_train_srcl import train_one_epoch_dual, evaluate,train_one_epoch_gan,evaluate_gan,train_one_epoch_cl
from discriminator_arch import UNetDiscriminatorSN,DiscriminatorForVGG

def get_args_parser():
    parser = argparse.ArgumentParser('ESRGAN fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--batch_number',default=2000,type = int )
    # Model parameters

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--use_focal_loss',action='store_true')
    # parser.add_argument('--pretrain_rrdb',default="/home/z.sun/super-resolution-medical/gfpgan/weights/best.pth",
    #                     help="use pretrained rrdb")
    parser.add_argument('--pretrain_rrdb',default="",
                        help="use pretrained rrdb")
    parser.add_argument('--discriminator_path',default="",
                        help="use pretrained discriminator")
    
    parser.add_argument('--finetune',default='',
                        help="fine-tuned model on 100k or 7k dataset")
    parser.add_argument('--linear_probe',action="store_true")

    parser.add_argument('--resnet50',action='store_true')
    parser.add_argument('--drop_out',default=0,type=float)
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR')
    

    # Augmentation parameters
    parser.add_argument('--balance_coeff',type=float,default=0.5)
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')


    # Dataset parameters
    parser.add_argument('--n_procs',default=48,type=int)
    parser.add_argument('--train_data_path', 
                        default="/home/data_repository/PATH-DT-MSU_dev/WSS2_v2_psi/train/", type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='data_v2.5', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=5, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default=1,type=int,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--train_gan',action='store_true')
    parser.add_argument('--runs_name',default='debug',type=str)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed',action="store_true")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    # Currently no need Distributed
    # misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.train_gan: 
        addtional_transform = A.Compose(
                transforms=[
                    # A.RandomResizedCrop(args.input_size, args.input_size),
                    A.HorizontalFlip(),
                    A.Rotate(limit=30),
                    # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ],
                additional_targets={'image1':'image'},
                is_check_shapes=False
            )
        transform = transforms.Compose(
        [
                transforms.Lambda(
                lambda x: (
                    addtional_transform(image = x[0],image1=x[1]),
                )
            ),
            transforms.Lambda(
                lambda x: (
                    x[0]['image'],
                    x[0]['image1'],
                )
            ),
            transforms.Lambda(
                lambda x: (
                    torch.from_numpy(x[0]).permute(2, 0, 1),
                    torch.from_numpy(x[1]).permute(2, 0, 1),
                )
            ),
        ]
        )
        from datasets.psi_dataset_random_sr import (
            PSIRandomSRDataset,
            PSIRandomSRDatasetParams,
        )
        ds_params_train = PSIRandomSRDatasetParams(
            path=Path(args.train_data_path),
            patch_size=args.input_size,
            layer_src=2 if args.input_size==224 else 4,
            layer_trg=1 if args.input_size==224 else 2,
        )
        dataset_train = TorchPSIDataset(
            ds_constructor=PSIRandomSRDataset,
            ds_params=ds_params_train,
            n_procs=args.n_procs,
            queue_size=2000,
            transform=transform,
            filter=low_entropy_filter(args.input_size),
        )
    else:
        addtional_transform = A.Compose(
        transforms=[
            # A.RandomResizedCrop(args.input_size, args.input_size),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(),
            A.Blur(blur_limit = 7,always_apply = False,p = 0.5 ),

            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
        additional_targets={'image1':'image'},
        is_check_shapes=False
    )
        transform = transforms.Compose(
        [
                transforms.Lambda(
                lambda x: (
                    addtional_transform(image = x[0],image1=x[1]),
                    x[2]
                )
            ),
            transforms.Lambda(
                lambda x: (
                    x[0]['image'],
                    x[0]['image1'],
                    x[1]
                )
            ),
            transforms.Lambda(
                lambda x: (
                    torch.from_numpy(x[0]).permute(2, 0, 1),
                    torch.from_numpy(x[1]).permute(2, 0, 1),
                    (torch.where(torch.from_numpy(x[2]) == 1 )[0]).squeeze()
                    
                    # (torch.where(torch.from_numpy(x[1]) == 1 )[0]).squeeze(),
                    # torch.from_numpy(x[1])
                )
            ),
        ]
        )
        if args.input_size==224:
            layer = 2
        elif args.input_size==112:
            layer = 4
        elif args.input_size==448:
            layer = 1 
        dataset_train = TorchPSIDataset(
                ds_constructor=PSIRegionSrClsDataset,
                ds_params=PSIRegionSrClsDatasetParams(
                    path=Path(args.train_data_path),
                    patch_size=args.input_size,
                    layer=layer,
                    region_intersection=0.7,
                    balance_coeff=args.balance_coeff,
                    hr_times=2 
                    # annotations_path=Path("/home/z.sun/wsi_SR_CL/annotation/WSS2_train/")
                ),
                n_procs=args.n_procs,
                queue_size=2000,
                transform=transform,
            ) 
    if args.input_size==224:
        lr_data_path = f"{args.val_data_path}/val_20x"
        hr_data_path = f"{args.val_data_path}/val_40x"
    elif args.input_size==112:
        lr_data_path = f"{args.val_data_path}/val_10x"
        hr_data_path = f"{args.val_data_path}/val_20x"
    elif args.input_size==448:
        return NotImplementedError
    dataset_val = SR_CLA_Dataset(lr_data_path=lr_data_path,hr_data_path=hr_data_path,is_train=False,args=args)
    if "NCT-CRC-HE-100K" in args.train_data_path or "CRC-VAL-HE-7K" in args.train_data_path:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # class_counts = {}
        # for label in dataset_train.hr_dataset.targets:
        #     if label in class_counts:
        #         class_counts[label] += 1
        #     else:
        #         class_counts[label] = 1
        # print(class_counts)
        # class_counts = list(class_counts.values())  # 每个类别的样本数量
        # print(class_counts)
        # # 计算每个类别的权重
        # class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

        # # 计算每个样本的权重
        # weights = class_weights[dataset_train.hr_dataset.targets]
        # sampler_train = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        # print("WeightedRandomSampler used")

    

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        save_dir = Path(f'{args.output_dir}/imgs')
        save_dir.mkdir(exist_ok=True, parents=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
            f.write(str(args))
    else:
        log_writer = None




    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=512,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if args.train_gan:
        discriminator = UNetDiscriminatorSN(3,64,True)
        # discriminator = DiscriminatorForVGG()
        loadnet = torch.load(args.discriminator_path,map_location='cpu')
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        elif 'params' in loadnet:
            keyname = 'params'
        else:
            keyname = 'model'
        info = discriminator.load_state_dict(loadnet[keyname], strict=True)
        print('load pretrained discrimiator',info)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.blr/4)
        discriminator.to(args.device)
        

        cri_perceptual = PerceptualLoss(layer_weights={'conv1_2': 0.1,
            'conv2_2': 0.1,
            'conv3_4': 1,
            'conv4_4': 1,
            'conv5_4': 1},vgg_type='vgg19',use_input_norm=True,perceptual_weight=1.0,style_weight=0,range_norm=False,criterion='l1').to(args.device)
    # val_list = [16221, 16200, 1300, 16209, 16674, 16429, 694, 16800, 16598, 16211, 16202, 736, 768, 17182, 17165, 16917, 16681, 1213, 16632, 16212, 16604, 16911, 16685, 16344, 16213, 16672, 16636, 1540, 16227, 1647, 16693, 1117, 16677, 16690, 16208, 1115, 16210, 16207, 16882, 16198, 16442, 16461, 16874, 16285, 16220, 1769, 16561, 861, 17285, 16312, 17050, 16635, 16936, 16587, 16349, 16761, 16840, 16644, 624, 16313, 16196, 16668, 16194, 866, 17137, 16552, 16684, 16437, 16348, 16665, 16239, 1281, 16944, 16601, 16206, 16645, 1962, 17035, 17073, 16455, 275, 16197, 16832, 16203, 1074, 962, 16199, 16634, 482, 16516, 17230, 16193, 17116, 16925, 432, 16638, 16214, 16378, 16205, 16195]
    #for v2 data 
    # val_list = [2613, 45365, 48121, 48466, 843, 48805, 4202, 1404, 45575, 2542, 2659, 49805, 45420, 46062, 52822, 2955, 46575, 48677, 52629, 49976, 1668, 48399, 4031, 48982, 51704, 46852, 46420, 45567, 51653, 47236, 671, 51286, 51728, 48879, 47326, 2735, 50805, 468, 811, 45493, 50527, 50833, 46279, 45558, 51041, 48273, 4265, 45410, 46391, 52596, 50996, 51602, 46380, 46103, 45889, 54251, 45829, 4984, 47135, 49010, 3462, 46414, 45475, 53606, 53138, 46505, 45873, 47050, 49657, 46010, 2612, 48717, 48740, 54236, 49970, 46261, 48617, 45484, 1768, 47399, 50189, 45629, 45761, 48480, 46115, 52575, 45657, 49825, 46046, 4062, 48276, 50490, 52572, 53607, 45895, 4554, 4999, 46127, 47226, 46572, 47446, 53629, 46754, 47476, 45951, 45361, 3385, 45379, 3306, 403, 1514, 1230, 3932, 45482, 47492, 49236, 46055, 3830, 53853, 49329, 681, 46337, 53354, 51072, 54722, 53706, 45622, 46498, 52908, 48710, 50761, 46136, 1569, 2569, 52929, 4229, 49547, 52757, 47331, 53524, 48584, 45542, 45710, 45599, 48771, 4178, 45565, 45415, 47747, 50308, 48016, 1577, 52957, 48149, 48176, 45117, 1717, 46621, 45936, 53229, 52599, 47184, 45516, 45676, 45164, 46543, 47982, 2094, 53435, 4329, 152, 48904, 45992, 1955, 49914, 50427, 48220, 2851, 45451, 882, 48907, 1748, 4324, 45903, 1110, 45566, 46274, 2298, 4542, 54208, 48631, 47214, 49427, 47275, 46367, 47477, 47481, 54271, 3373, 45840, 53536, 47217, 3657, 54445, 48427, 53310, 45314, 46433, 52559, 46870, 457, 3630, 49187, 377, 48726, 2356, 51386, 3992, 46075, 49291, 48138, 45241, 48491, 45439, 4772, 52381, 4963, 4711, 48605, 45236, 54388, 52328, 46052, 48342, 47099, 48319, 45501, 45843, 51792, 52334, 49633, 51443, 52333, 3262, 47601, 1393, 49872, 325, 52325, 49411, 1902, 49875, 48837, 165, 45491, 47391, 1541, 485, 52906, 4920, 45187, 3054, 46483, 1981, 47799, 45183, 47480, 46566, 614, 4496, 3098, 1034, 45458, 2184, 46002, 46311, 4564, 46049, 46292, 48212, 50, 47420, 3725, 46587, 51348, 47354, 49191, 47149, 53469, 46196, 45883, 48994, 2404, 45169, 49616, 46748, 46644, 45435, 49180, 47159, 48869, 45382, 53677, 50631, 50515, 4851, 52573, 3067, 54279, 50147, 53391, 46226, 49500, 50121, 45867, 48917, 45557, 2737, 53491, 46354, 525, 51572, 1893, 49000, 49294, 4, 2853, 45775, 45799, 45638, 46271, 46070, 45165, 878, 52962, 48382, 52207, 2481, 51098, 1178, 49127, 45450, 46278, 49612, 54079, 46198, 50082, 48626, 53421, 354, 46608, 3492, 3164, 4838, 47453, 52448, 1882, 1808, 2839, 2464, 347, 2753, 49373, 45966, 50410, 45209, 53043, 50245, 1134, 45340, 54108, 45536, 48594, 49197, 593, 45682, 50695, 49450, 49117, 46831, 50717, 4013, 46015, 53762, 52046, 45198, 53916, 47936, 54629, 45611, 2018, 48578, 52490, 51320, 642, 46308, 49484, 48761, 51498, 54300]
    #for v2.5 data 
    val_list = [2379, 52342, 51635, 44272, 2277, 46325, 43608, 44287, 52443, 3511, 51639, 43811, 43791, 44767, 50014, 44032, 518, 4416, 46619, 51144, 48141, 1457, 2434, 43939, 44836, 47643, 49254, 50561, 52421, 3380, 46987, 43617, 4158, 45926, 2292, 46942, 4400, 766, 43795, 3937, 43542, 2393, 660, 3137, 47850, 44382, 44848, 1754, 44149, 4254, 47486, 44641, 49858, 47082, 48827, 44580, 47489, 52873, 4276, 52615, 43669, 45635, 1081, 44126, 3430, 52536, 51348, 43880, 44706, 1039, 49444, 2618, 48427, 43901, 1072, 46083, 50880, 45085, 50597, 50326, 1534, 47107, 44532, 2667, 46930, 46308, 2329, 43710, 44172, 52495, 48826, 47682, 47347, 44384, 52234, 45859, 44184, 1783, 1713, 49870, 43823, 43797, 43979, 43956, 47461, 49472, 52332, 4607, 43752, 45253, 4740, 44030, 3810, 43855, 44096, 45421, 52577, 45203, 50196, 47858, 44573, 48057, 640, 47211, 2780, 27, 47479, 52209, 52671, 48195, 44439, 43818, 51244, 3700, 43718, 46187, 2568, 3453, 45313, 2322, 49839, 4694, 44129, 44578, 43591, 50381, 181, 847, 44733, 51684, 47361, 46669, 45260, 1753, 2626, 47714, 44052, 50516, 52646, 48502, 1829, 654, 47630, 46107, 49809, 49309, 44281, 47131, 45529, 44034, 44185, 52546, 44627, 47729, 43636, 50603, 44663, 48501, 43547, 46405, 3193, 46751, 4241, 704, 43815, 52661, 50710, 52801, 538, 46818, 45557, 44530, 3476, 2882, 44173, 50456, 49346, 4086, 49148, 46099, 47536, 45798, 45137, 3298, 4373, 145, 45172, 43917, 52440, 45734, 2475, 50866, 46458, 52749, 43587, 51181, 3034, 52374, 44642, 47656, 44102, 46162, 46948, 547, 44156, 46222, 44555, 44620, 45151, 2059, 43786, 1153, 45692, 705, 47388, 46664, 1617, 43696, 45374, 51500, 47055, 2445, 46899, 458, 1353, 44001, 45853, 44695, 44699, 43804, 44579, 202, 48130, 46520, 45625, 3720, 745, 48198, 49506, 51394, 3175, 50369, 43783, 44477, 44346, 49487, 3787, 43936, 50222, 3032, 49243, 47836, 51990, 1123, 51723, 43834, 44770, 44088, 2394, 44907, 47507, 47475, 2075, 45414, 48194, 49857, 45314, 43543, 52126, 48053, 2144, 47139, 52553, 43973, 2859, 45659, 46251, 44300, 44271, 44517, 49158, 45643, 46656, 2463, 2060, 43573, 47494, 44159, 45944, 51395, 49622, 43473, 47892, 52364, 52901, 45894, 47993, 2790, 47372, 4714, 45328, 51280, 43599, 44368, 47725, 43829, 43520, 47695, 43789, 44691, 46363, 47110, 44371, 49497, 44018, 43642, 1066, 48670, 43579, 1814, 52515, 46891, 2512, 4429, 47106, 44471, 2356, 2935, 1418, 373, 47906, 2617, 49353, 48477, 51382, 46780, 199, 44819, 49997, 46744, 46672, 49600, 44487, 1127, 52734, 43733, 50886, 44917, 45955, 3789, 46898, 51177, 43559, 1174, 46243, 45808, 45592, 46100, 48377, 49993, 3385, 45550, 4260, 44689, 52279, 3008, 46680, 51994, 46832, 2397, 44795, 51306, 3904, 50035, 44859, 525, 48361, 45297, 47669, 49389]

    if args.resnet50:
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(args.drop_out),
            torch.nn.Linear(num_ftrs, args.nb_classes)
        )
    else:
        model = models_esrc(num_in_ch=3,num_feat=64, num_block=15, num_grow_ch=32, scale=2,drop_rate=args.drop_out,
                        num_classes=args.nb_classes)
    if args.finetune != "" and not args.resnet50:
        loadnet = torch.load(args.finetune, map_location=torch.device('cpu'))['model']
        loadnet.pop('head.0.weight', None)
        loadnet.pop('head.0.bias', None)
        loadnet.pop('head.2.bias', None)
        loadnet.pop('head.2.weight', None)
        info = model.load_state_dict(loadnet, strict=False)
        print('load pretrained fine-tuned encoder',info)
    elif args.pretrain_rrdb != "" and not args.resnet50:
        loadnet = torch.load(args.pretrain_rrdb, map_location=torch.device('cpu'))

        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        elif 'params' in loadnet:
            keyname = 'params'
        else:
            keyname = 'model'
        keys_to_remove = []
        for name in loadnet[keyname].keys():
            if name in ["head.0.weight", "head.0.bias", "head.1.weight", "head.1.bias"]:
                keys_to_remove.append(name)

        # Remove the keys from the dictionary
        for name in keys_to_remove:
            if name in loadnet[keyname]:
                print(f"remove {name} from ckpt")
                del loadnet[keyname][name]

        info = model.load_state_dict(loadnet[keyname], strict=False)
        print('load pretrained encoder',info)
    model.to(device)

    model_without_ddp = model

    if args.linear_probe:
        for param in model_without_ddp.parameters():
            param.requires_grad = False
        for param in model_without_ddp.head.parameters():
            param.requires_grad = True
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    if args.resnet50:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    else:
        # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        #     no_weight_decay_list=[],
        #     layer_decay=args.layer_decay
        # )
        # param_groups = model.parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.blr)
        # optimizer = torch.optim.AdamW(param_groups, lr=args.blr)
    #
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.use_focal_loss:
        criterion = misc.focal_loss(alpha=[0.2,0.2,0.2,0.2,0.2],
                        gamma=2,
                        reduction='mean',
                        device=device,
                        dtype=torch.float32
                    )
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    # test_stats = evaluate(data_loader_val, model, device
    for epoch in range(args.start_epoch, args.epochs):
        args.epoch = epoch 
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if args.train_gan:    
            test_stats = evaluate_gan(dataset_val, model, device,args,val_list=val_list)
            if log_writer is not None:
                log_writer.add_scalar('val/psnr', test_stats['psnr'], epoch)
                log_writer.add_scalar('val/ssim', test_stats['ssim'], epoch)    
            train_stats = train_one_epoch_gan(
                model,discriminator, cri_perceptual,optimizer_d, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )   
            
            if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                        )
        else :
            
            test_stats = evaluate_gan(dataset_val, model, device,args,val_list=val_list)
            
            if log_writer is not None:
                log_writer.add_scalar('val/psnr', test_stats['psnr'], epoch)
                log_writer.add_scalar('val/ssim', test_stats['ssim'], epoch)
            train_stats = train_one_epoch_dual(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )


            test_stats,class_accuracy = evaluate(data_loader_val, model, device,args)
            # print('------------------------test on train----------------------------\n')
            # evaluate(data_loader_val_2, model, device,args)
            # print('------------------------test on train finish----------------------------\n')
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max(max_accuracy, test_stats["acc1"]) > max_accuracy :
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch,model_name = 'best')
            else:
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch
                        )
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc3', test_stats['acc3'], epoch)
                log_writer.add_scalar('perf/test_loss_sr', test_stats['loss_sr'], epoch)
                log_writer.add_scalar('perf/test_loss_cl', test_stats['loss_cl'], epoch)
                for class_label, acc in class_accuracy.items():
                    log_writer.add_scalar(f'class_accuracy/{class_label}', acc['acc'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.distributed:
        pass
    else :
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        args.device = 0
    if args.output_dir:
        file_path = os.path.abspath(__file__)
        
        args.output_dir = str(Path.joinpath(Path(__file__).parent,args.output_dir,args.runs_name))
        args.log_dir = args.output_dir
        # print(f"Current time: {current_time}")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)