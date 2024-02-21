import math
import sys
from typing import Iterable, Optional
import numpy as np 
import torch
import pandas as pd 
import dataframe_image as dfi 
from timm.data import Mixup
from timm.utils import accuracy
from skimage.metrics import structural_similarity
from pathlib import Path
import misc
import lr_sched
import cv2 
from tqdm import tqdm 
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import L1Loss

cri_gan = GANLoss(gan_type='vanilla',real_label_val=1.0,fake_label_val=0,loss_weight=0.1)
cri_pix = L1Loss()
def train_one_epoch_gan(model: torch.nn.Module, net_d:torch.nn.Module,
                        cri_perceptual: torch.nn.Module,
                        optimizer_d:torch.optim.Optimizer,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    torch.cuda.empty_cache()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_d.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (lr_imgs,hr_imgs) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if data_iter_step == args.batch_number:
            break
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)


        l1_gt = hr_imgs
        percep_gt = hr_imgs
        gan_gt = hr_imgs
        # optimize net_g
        for p in net_d.parameters():
            p.requires_grad = False

        with torch.cuda.amp.autocast():
            outputs = model(lr_imgs,mode='sr')
            output = outputs
            l_g_pix = cri_pix(output, l1_gt)
            l_g_percep, l_g_style = cri_perceptual(output, percep_gt)
            fake_g_pred = net_d(output)
            l_g_gan = cri_gan(fake_g_pred, True, is_disc=False)

            loss = l_g_pix + l_g_percep + l_g_gan

            

        loss_value = loss.item()
        loss_sr_value = (l_g_pix+l_g_percep).item()
        loss_g_gan_value = l_g_gan.item() 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        # optimize net_d
        for p in net_d.parameters():
            p.requires_grad = True
        with torch.cuda.amp.autocast():
            real_d_pred = net_d(gan_gt)
            l_d_real = cri_gan(real_d_pred, True, is_disc=True)
  
            l_d_real.backward()
            # fake
            fake_d_pred = net_d(output.detach().clone())  # clone for pt1.9
            l_d_fake = cri_gan(fake_d_pred, False, is_disc=True)

            l_d_fake.backward()
        optimizer_d.step()
        loss_l_d_fake_value = l_d_fake.item() 
        loss_l_d_real_value = l_d_real.item() 
        #optimize net_d
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            optimizer_d.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_sr=loss_sr_value)
        metric_logger.update(loss_g_gan=loss_g_gan_value)
        metric_logger.update(loss_l_d_fake=loss_l_d_fake_value)
        metric_logger.update(loss_l_d_real=loss_l_d_real_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_sr_value_reduce = misc.all_reduce_mean(loss_sr_value)
        loss_g_gan_value_reduce = misc.all_reduce_mean(loss_g_gan_value)
        loss_l_d_fake_value_reduce = misc.all_reduce_mean(loss_l_d_fake_value)
        loss_l_d_real_value_reduce = misc.all_reduce_mean(loss_l_d_real_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_sr', loss_sr_value_reduce, epoch_1000x)
            log_writer.add_scalar('ganloss/loss_g_gan', loss_g_gan_value_reduce, epoch_1000x)
            log_writer.add_scalar('ganloss/loss_l_d_fake', loss_l_d_fake_value_reduce, epoch_1000x)
            log_writer.add_scalar('ganloss/loss_l_d_real', loss_l_d_real_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_cl(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    torch.cuda.empty_cache()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (lr_imgs,hr_imgs, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # if data_iter_step == args.batch_number:
        #     break
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            lr_imgs, targets = mixup_fn(lr_imgs, targets)

        with torch.cuda.amp.autocast():
            outputs = model(lr_imgs,mode = 'cl')
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_dual(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    torch.cuda.empty_cache()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (lr_imgs,hr_imgs, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # if data_iter_step == args.batch_number:
        #     break
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            lr_imgs, targets = mixup_fn(lr_imgs, targets)

        with torch.cuda.amp.autocast():
            outputs = model(lr_imgs)
            logits,out = outputs
            loss_cl = criterion(logits, targets)
            loss_sr = torch.nn.L1Loss()(out,hr_imgs)
            loss = loss_cl + loss_sr 

        loss_value = loss.item()
        loss_sr_value = loss_sr.item()
        loss_cl_value = loss_cl.item() 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_sr=loss_sr_value)
        metric_logger.update(loss_cl=loss_cl_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_sr_value_reduce = misc.all_reduce_mean(loss_sr_value)
        loss_cl_value_reduce = misc.all_reduce_mean(loss_cl_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_sr', loss_sr_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_cl', loss_cl_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device,args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    class_accuracy = {}
    labels = ['AT', 'BG','LP', 'MM', 'TUM']

    # 初始化行和列


    # 创建 DataFrame
    df = pd.DataFrame(np.zeros((labels.__len__(),labels.__len__()),int),columns=labels, index=labels)

    # 设置标题
    df.columns.name = 'gt'
    df.index.name = 'pred'
    for data_iter_step,(lr_imgs,hr_imgs, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # if data_iter_step == 20:
        #     break
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(lr_imgs)
            if isinstance(outputs,tuple):
                output,out = outputs
                loss_cl = criterion(output, targets)
                loss_sr = torch.nn.L1Loss()(out,hr_imgs)
            else:
                output = outputs
                loss_cl = criterion(output, targets)
                loss_sr = torch.tensor([1e-9])
            

        acc1, acc3 = accuracy(output, targets, topk=(1, 3))
        _, predicted = torch.max(output, 1)

        # 遍历预测结果和真实标签，计算每个类别的准确率
        for pred, tar in zip(predicted, targets):
            pred = pred.item()
            tar = tar.item()
            df.iloc[pred,tar] += 1
            if tar not in class_accuracy:
                class_accuracy[tar] = {'total': 0, 'correct': 0}

            class_accuracy[tar]['total'] += 1
            if pred == tar:
                class_accuracy[tar]['correct'] += 1

    # 计算每个类别的准确率


        batch_size = lr_imgs.shape[0]
        metric_logger.update(loss_cl=loss_cl.item())
        metric_logger.update(loss_sr=loss_sr.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
    # gather the stats from all processes
    dfi.export(df.style.background_gradient(), f"{args.output_dir}/imgs/matrix_conf_{args.epoch}.png", table_conversion="matplotlib")
    # eval_sr_img("/home/z.sun/wsi_SR_CL/test_img/TUM/org_1.jpg",model,device,f'{args.output_dir}/out_{args.epoch}.png')
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} loss_cl {losses.global_avg:.3f}, loss_sr {loss_sr.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, losses=metric_logger.loss_cl,loss_sr=metric_logger.loss_sr))
    for class_label, acc in class_accuracy.items():
            acc['acc'] = acc['correct'] / acc['total'] * 100
    print(class_accuracy)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},class_accuracy


@torch.no_grad()
def evaluate_gan(val_dataset, model, device,args,val_list):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    save_dir = Path(f'{args.output_dir}/val_hr_imgs_{args.epoch}')
    save_dir.mkdir(exist_ok=True, parents=True)
    k=0
    # switch to evaluation mode
    model.eval()
    for i in tqdm(val_list):
        lr_img,hr_img,label_hr = val_dataset.__getitem__(i)
        lr_img = lr_img.to(device)
        filename = val_dataset.hr_dataset.imgs[i][0].split('/')[-1]
        with torch.no_grad():
            out = model(lr_img[None,:,:,:],mode='sr')
        
        output_img = out.data.squeeze().float().detach().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
        output = (output_img * 255.0).round().astype(np.uint8)
        hr_img = np.transpose(hr_img.numpy()[[2, 1, 0], :, :], (1, 2, 0))
        hr_img = ( hr_img * 255.0).round().astype(np.uint8)
        psnr = cv2.PSNR(output,hr_img)
        ssim = structural_similarity(output,hr_img,channel_axis=2)
        metric_logger.meters['psnr'].update(psnr, n=1)
        metric_logger.meters['ssim'].update(ssim, n=1)
        k=k+1
        if k %20 ==0 :
            cv2.imwrite(str(save_dir/filename), output)
    print('* psnr {psnr.global_avg:.3f} ssim {ssim.global_avg:.3f}'
          .format(psnr=metric_logger.psnr, ssim=metric_logger.ssim))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        
 

def eval_sr_img(img_path,model,device,save_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = img.astype(np.float32)
    if np.max(img) > 256:  # 16-bit image
        max_range = 65535
        print('\tInput is a 16-bit image')
    else:
        max_range = 255
    img = img / max_range
    if len(img.shape) == 2:  # gray image
        img_mode = 'L'
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA image with alpha channel
        img_mode = 'RGBA'
        alpha = img[:, :, 3]
        img = img[:, :, 0:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
    else:
        img_mode = 'RGB'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img = img.to(device)
    with torch.no_grad():
        logits,out = model(img[None,:,:,:])
        output_img = out.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))
    output = (output_img * 255.0).round().astype(np.uint8)
    cv2.imwrite(save_path, output)