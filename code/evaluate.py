# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import argparse
import json
from skimage.metrics import hausdorff_distance
import torch

import numpy as np
import math
from spectr import SpecTr

from torch.utils.data import DataLoader
import os
from local_utils.seed_everything import seed_reproducer
import pandas as pd

from tqdm import tqdm
from Data_Generate import Data_Generate_Cho
from local_utils.metrics import iou, dice


def main(args):
    seed_reproducer(42)
    root_path = args.root_path
    dataset_divide = args.dataset_divide
    experiment_name = args.experiment_name
    checkpoint = args.checkpoint
    worker = args.worker
    device = args.device
    batch = args.batch

    model_path = os.path.join(checkpoint, experiment_name)

    images_root_path = os.path.join(root_path, args.dataset_hyper)
    mask_root_path = os.path.join(root_path, args.dataset_mask)
    dataset_json = os.path.join(root_path, dataset_divide)
    with open(dataset_json, 'r') as load_f:
        dataset_dict = json.load(load_f)

    device = torch.device(device)
    channels = args.channels_index
    spectral_number = args.spectral_number if channels is None else len(args.channels_index)
    multi_class = 1

    labels, outs = [], []
    model = SpecTr(choose_translayer=args.choose_translayer,
                           spatial_size=(args.cutting, args.cutting),
                           max_seq=spectral_number,
                           classes=multi_class,
                           decode_choice=args.decode_choice,
                           init_values=args.init_values)
    model = model.to(device)

    history = {'val_iou': [], 'val_dice': [], 'val_haus': []}
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

    for k in args.fold:
        val_files = dataset_dict[f'fold{k}']
        val_images_path = [os.path.join(images_root_path, i) for i in dataset_dict[f'fold{k}']]
        val_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in dataset_dict[f'fold{k}']]
        print(f'the number of valfiles is {len(val_files)}')
        val_db = Data_Generate_Cho(val_images_path, val_masks_path, cutting=None, transform=None,
                                          channels=channels, outtype=args.outtype)
        val_loader = DataLoader(val_db, batch_size=1, shuffle=False, num_workers=worker)

        model.load_state_dict(torch.load(os.path.join(model_path, f'final_{k}fold_74.pth'),
                                         map_location=lambda storage, loc: storage.cuda()))

        print('now start evaluate ...')
        model.eval()
        for idx, sample in enumerate(tqdm(val_loader)):
            image, label = sample
            image = image.squeeze()
            spectrum_shape, shape_h, shape_w = image.shape
            patch_idx = list(patch_index((spectrum_shape, shape_h, shape_w), (args.cutting, args.cutting, spectrum_shape),
                                         (64, 128, 1)))  # origan shape is 256, 320; 128=320-192, 64=256-192
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
            labels.extend(label)
            outs.extend(out)

    val_iou = np.array([iou(o[0], l[0]) for l, o in zip(labels, outs)])
    val_dice = np.array([dice(o[0], l[0]) for l, o in zip(labels, outs)])
    val_haus = np.array([hausdorff_distance(o[0], l[0]) for l, o in zip(labels, outs)]) #if
                         # hausdorff_distance(o[0], l[0]) != float('inf')])

    history['val_iou'].append(val_iou.mean())
    history['val_dice'].append(val_dice.mean())
    history['val_haus'].append(val_haus.mean())
    history['val_iou'].append(val_iou.std())
    history['val_dice'].append(val_dice.std())
    history['val_haus'].append(val_haus.std())

    print(f"the valid dataset iou & dice & hausdorff_distance is {val_iou.mean()} & {val_dice.mean()} & {val_haus.mean()}")
    history_pd = pd.DataFrame(history)
    history_pd.to_csv(os.path.join(model_path, f'metric.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', '-r', type=str, default='./Cholangiocarcinoma/L')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='four_fold.json')
    parser.add_argument('--device', '-dev', type=str, default='cuda:0')
    parser.add_argument('--fold', '-fold', type=int, default=[1, 2, 3, 4], nargs='+')

    parser.add_argument('--spectral_number', '-sn', default=60, type=int)
    parser.add_argument('--channels_index', '-chi', type=int, default=None, nargs='+')
    parser.add_argument('--worker', '-nw', type=int,
                        default=4)
    parser.add_argument('--batch', '-b', type=int, default=1)
    parser.add_argument('--outtype', '-outt', type=str,
                        default='3d')
    parser.add_argument('--checkpoint', '-o', type=str, default='checkpoint')
    parser.add_argument('--experiment_name', '-name', type=str,
                        default='SpecTr_XXX')
    parser.add_argument('--choose_translayer', '-ct', nargs='+', type=int, default=[0, 1, 1, 1])
    parser.add_argument('--cutting', '-cut', default=192, type=int)
    parser.add_argument('--epochs', '-e', type=int, default=75)
    parser.add_argument('--decode_choice', '-dc', default='3D', choices=['3D', 'decoder_2D'])
    parser.add_argument('--init_values', '-initv', type=float, default=0.01)

    args = parser.parse_args()

    main(args)
