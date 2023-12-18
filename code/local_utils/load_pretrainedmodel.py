#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import torch.nn as nn
import torch

def load_pretrainedmodel(loaded_model, pretrainedmodel_path):
    
    save_model = torch.load(pretrainedmodel_path)
    model_dict =  loaded_model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if 'encoders' in k}
    model_dict.update(state_dict)
    loaded_model.load_state_dict(model_dict)
    return loaded_model

def load_gvt_pretrainedmodel(loaded_model, pretrainedmodel_path):
    
    save_model = torch.load(pretrainedmodel_path)
    model_dict =  loaded_model.state_dict()
    state_dict = {k:v for idx,(k,v) in enumerate(save_model.items()) if 'head' not in k}
    model_dict.update(state_dict)
    loaded_model.load_state_dict(model_dict)
    return loaded_model

def load_pvt_pretrainedmodel(loaded_model, pretrainedmodel_path, pretrained_pos=False):
    save_model = torch.load(pretrainedmodel_path)
    model_dict =  loaded_model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if ('cls_token' not in k) and ('head' not in k)}
    state_dict = {f'encoder.{k}':v for k,v in state_dict.items()}
    print(state_dict.keys())
    model_dict.update(state_dict)
    print(f"success load {len(state_dict.keys())} parameters")

    loaded_model.load_state_dict(model_dict)
    
    return loaded_model
def load_swin_pretrainedmodel(loaded_model,pretrainedmodel_path):
    save_model = (torch.load(pretrainedmodel_path))
    state_dict = {k:v for k,v in save_model['state_dict'].items() if ('backbone' in k)}
    state_dict = {f'{k[9:]}':v for k,v in state_dict.items()}
    loaded_model.encoder.load_state_dict(state_dict)
    return loaded_model
#def load_transformer_pretrainedmodel(loaded_model, pretrainedmodel_path, pretrained_pos=False):
#    save_model = torch.load(pretrainedmodel_path)
#    model_dict =  loaded_model.state_dict()
#    if pretrained_pos:
#        save_model['pos_embed'] = save_model['pos_embed'].resize_(1,10,768)# get [:9] from [:16] sequence length
#        state_dict = {k:v for k,v in save_model.items() if k in model_dict}
#    else:
#        state_dict = {k:v for k,v in save_model.items() if k in model_dict and k not in 'pos_embed'}
#        
#    print(f"success load {len(state_dict.keys())} parameters")
#    model_dict.update(state_dict)
#    loaded_model.load_state_dict(model_dict)
#    
#    return loaded_model

if __name__ == '__main__':
    x = torch.load('/home/ubuntu/T/redhouse/utils/attention_3d/\
redhouse-checkpoint-new/forpretrained_3dUnet_03class_0117/best_forpretrained_3dUnet_03class_0117.pth')
    for k,v in x.items():
        print(k)
#    
