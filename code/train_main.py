#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import os
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp

from spectr import SpecTr
from torch import optim
from torch.utils.data import DataLoader

from local_utils.tools import save_dict
from local_utils.seed_everything import seed_reproducer

from tqdm import tqdm
from Data_Generate import Data_Generate_Cho
from argument import Transform
from local_utils.misc import AverageMeter
from local_utils.tools import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from local_utils.metrics import iou, dice

import warnings
warnings.filterwarnings('ignore')


def main(args):
    seed_reproducer(42)

    root_path = args.root_path
    dataset_hyper = args.dataset_hyper
    dataset_mask = args.dataset_mask
    dataset_divide = args.dataset_divide
    batch = args.batch
    lr = args.lr
    wd = args.wd
    experiment_name = args.experiment_name
    output_path = args.output
    epochs = args.epochs
    cutting = args.cutting
    spectral_number = args.spectral_number
    fold = args.fold
    choose_translayer = args.choose_translayer
    worker = args.worker
    outtype = args.outtype
    channels_index = args.channels_index
    device = args.device
    decode_choice = args.decode_choice
    init_values = args.init_values

    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)
    dataset_json = os.path.join(root_path, dataset_divide)
    with open(dataset_json, 'r') as load_f:
        dataset_dict = json.load(load_f)

    #Data Augmentation
    transform = Transform(Rotate_ratio=0.2, Flip_ratio=0.2)
    device = torch.device(device)

    if os.path.exists(f'{output_path}/{experiment_name}') == False:
        os.mkdir(f'{output_path}/{experiment_name}')
    save_dict(os.path.join(f'{output_path}/{experiment_name}', 'args.csv'), args.__dict__)

    channels = channels_index
    spectral_number = spectral_number if channels is None else len(channels_index)
    multi_class = 1

    dice_criterion = smp.losses.DiceLoss(eps=1., mode='binary', from_logits=False)
    bce_criterion = nn.BCELoss()

    Miou = iou
    MDice = dice

    #For slide window operation in the validation stage
    def patch_index(shape, patchsize, stride):
        s, h, w = shape
        sx = (w - patchsize[1]) // stride[1] + 1
        sy = (h - patchsize[0]) // stride[0] + 1
        sz = (s - patchsize[2]) // stride[2] + 1

        for x in range(sx):
            xs = stride[1] * x
            for y in range(sy):
                ys = stride[0] * y
                for z in range(sz):
                    zs = stride[2] * z
                    yield slice(zs, zs + patchsize[2]), slice(ys, ys + patchsize[0]), slice(xs, xs + patchsize[1])


    for k in fold:
        train_fold = list(set([1, 2, 3, 4]) - set([k]))
        print(f"train_fold is {train_fold} and valid_fold is {k}")

        train_file_dict = dataset_dict[f'fold{train_fold[0]}'] + dataset_dict[f'fold{train_fold[1]}'] + dataset_dict[
            f'fold{train_fold[2]}']

        train_images_path = [os.path.join(images_root_path, i) for i in train_file_dict]
        train_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in train_file_dict]
        val_images_path = [os.path.join(images_root_path, i) for i in dataset_dict[f'fold{k}']]
        val_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in dataset_dict[f'fold{k}']]

        train_db = Data_Generate_Cho(train_images_path, train_masks_path, cutting=cutting,
                                            transform=transform, channels=channels, outtype=outtype)
        train_loader = DataLoader(train_db, batch_size=batch, shuffle=True, num_workers=worker)

        val_db = Data_Generate_Cho(val_images_path, val_masks_path, cutting=None, transform=None,
                                          channels=channels, outtype=outtype)
        val_loader = DataLoader(val_db, batch_size=1, shuffle=False, num_workers=worker)

        model = SpecTr(choose_translayer=choose_translayer,
                           spatial_size=(cutting, cutting),
                           max_seq=spectral_number,
                           classes=multi_class,
                           decode_choice=decode_choice,
                           init_values=init_values).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-8)

        # only record, we are not use early stop.
        early_stopping_val = EarlyStopping(patience=1000, verbose=True,
                                           path=os.path.join(f'{output_path}/{experiment_name}',
                                                             f'best_fold{k}_{experiment_name}.pth'))

        history = {'epoch': [], 'LR': [], 'train_loss': [], 'train_iou': [], 'val_dice': [], 'val_iou': [],
                   'val_count': []}


        for epoch in range(epochs):
            train_losses = AverageMeter()
            val_losses = AverageMeter()
            train_iou, val_iou, val_dice = 0, 0, 0
            print('now start train ..')
            print('epoch {}/{}, LR:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
            train_losses.reset()
            model.train()
            try:
                for idx, sample in enumerate(tqdm(train_loader)):
                    image, label = sample
                    image, label = image.to(device), label.to(device)
                    out = model(image)
                    loss = dice_criterion(out, label) * 0.5 + bce_criterion(out, label) * 0.5

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.update(loss.item())
                    out = out.cpu().detach().numpy()
                    label = label.cpu().detach().numpy()
                    out = np.where(out > 0.5, 1, 0)
                    label = np.where(label > 0.5, 1, 0)

                    train_iou = train_iou + np.mean(
                        [Miou(out[b], label[b]) for b in range(len(out))])

                train_iou = train_iou / (idx + 1)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, please reduce batch')
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return
                else:
                    raise e

            print('now start evaluate ...')
            model.eval()
            val_losses.reset()
            for idx, sample in enumerate(tqdm(val_loader)):
                image, label = sample
                image = image.squeeze()
                spectrum_shape, shape_h, shape_w = image.shape
                patch_idx = list(patch_index((spectrum_shape, shape_h, shape_w), (cutting, cutting, spectrum_shape),
                                             (64, 128, 1))) # origan shape is 256, 320; 128=320-192, 64=256-192
                num_collect = torch.zeros((shape_h, shape_w), dtype=torch.uint8).to(device)
                pred_collect = torch.zeros((shape_h, shape_w)).to(device)
                for i in range(0, len(patch_idx), batch):
                    with torch.no_grad():
                        output = model(torch.stack([image[x] for x in patch_idx[i:i + batch]])[None].to(device)).squeeze(1)
                    for j in range(output.size(0)):
                       num_collect[patch_idx[i + j][1:]] += 1
                       pred_collect[patch_idx[i + j][1:]] += output[j]

                out = pred_collect / num_collect.float()
                out[torch.isnan(out)] = 0

                out, label = out.cpu().detach().numpy()[None][None], label.cpu().detach().numpy()

                out = np.where(out > 0.5, 1, 0)
                label = np.where(label > 0.5, 1, 0)
                val_dice = val_dice + MDice(out, label)
                val_iou = val_iou + Miou(out, label)

            val_iou = val_iou / (idx + 1)
            val_dice = val_dice / (idx + 1)

            print('epoch {}/{}\t LR:{}\t train loss:{}\t train_iou:{}\t val_dice:{}\t val_iou:{}' \
                  .format(epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_losses.avg, train_iou, val_dice,
                          val_iou))
            history['train_loss'].append(train_losses.avg)
            history['val_dice'].append(val_dice)
            history['val_iou'].append(val_iou)
            history['train_iou'].append(train_iou)

            history['epoch'].append(epoch + 1)
            history['LR'].append(optimizer.param_groups[0]['lr'])

            scheduler.step()
            early_stopping_val(-val_dice, model)
            history['val_count'].append(early_stopping_val.counter)

            if args.save_every_epoch:
                if (epoch + 1) % 5 == 0:
                    torch.save(model.state_dict(),
                               os.path.join(f'{output_path}/{experiment_name}', f'middle_{k}fold_{epoch}.pth'))

            if epoch + 1 == epochs:
                torch.save(model.state_dict(),
                           os.path.join(f'{output_path}/{experiment_name}', f'final_{k}fold_{epoch}.pth'))


            if early_stopping_val.early_stop:
                print("Early stopping")
                break

            history_pd = pd.DataFrame(history)
            history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log_fold{k}.csv'), index=False)
        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log_fold{k}.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', '-r', type=str, default='./Cholangiocarcinoma/L')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='four_fold.json')
    parser.add_argument('--fold', '-fold', type=int, default=[1, 2, 3, 4], nargs='+')
    parser.add_argument('--device', '-dev', type=str, default='cuda:0')

    parser.add_argument('--worker', '-nw', type=int,
                        default=4)
    parser.add_argument('--outtype', '-outt', type=str,
                        default='3d')

    parser.add_argument('--batch', '-b', type=int, default=1)

    parser.add_argument('--lr', '-l', default=0.0003, type=float)
    parser.add_argument('--wd', '-w', default=5e-4, type=float)

    parser.add_argument('--spectral_number', '-sn', default=60, type=int)
    parser.add_argument('--channels_index', '-chi', type=int, default=None, nargs='+')

    parser.add_argument('--output', '-o', type=str, default='./checkpoint')
    parser.add_argument('--choose_translayer', '-ct', nargs='+', type=int, default=[0, 1, 1, 1])
    parser.add_argument('--experiment_name', '-name', type=str, default='SpecTr_XXXX')
    parser.add_argument('--cutting', '-cut', default=192, type=int)
    parser.add_argument('--epochs', '-e', type=int, default=75)
    parser.add_argument('--decode_choice', '-dc', default='3D', choices=['3D', 'decoder_2D'])
    parser.add_argument('--init_values', '-initv', type=float, default=0.01)
    parser.add_argument('--save_every_epoch', '-see', default=False, action='store_true')

    args = parser.parse_args()
    main(args)
