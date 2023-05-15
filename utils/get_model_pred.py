import gc
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
from pathlib import Path
import math
import json


class Classifier(nn.Module):
    def __init__(self , nb_classes):
        super(Classifier, self).__init__()
        
        self.ball_prefc = nn.LazyLinear(256)
        self.ball_attention = nn.TransformerEncoderLayer(256 , 1 , dim_feedforward = 512 , dropout=0.15 , batch_first = True)
        self.ball_fc = nn.LazyLinear(nb_classes)
        
    def forward(self , ball):
        
        ballout = self.ball_prefc(ball)
        ballout = self.ball_attention(ballout)
        ballout = self.ball_fc(ballout)

        return ballout
def build_model():    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nb_classes = 2
    model = Classifier(nb_classes).to(device)
    _exp_name = "hit_model"
    read_model = "./models/0514_hit_model_best_smooth.ckpt"

    model.load_state_dict(torch.load(read_model))

    model.eval()
    return model

def get_model_pred(model , ball_frame ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f_range = 15

    rel_li = []
    pred_dic = {}
    
    idx = -f_range 
    # for _ in range(-f_range + 1 , len(ball_frame) + f_range):
    for _ , j in ball_frame.iterrows():
        # print(idx)

        while idx < j['Frame']:
            rel_li += [0,0,0]
            idx += 1

        if idx == j['Frame']: 
            rel_li += [int(j['Visibility']),int(j['X']),int(j['Y'])]
            idx += 1

        if len(rel_li) == (f_range * 2 + 1) * 3:
            ball = torch.FloatTensor(rel_li)
            # print(ball)
            pred = model(ball.unsqueeze(0).to(device))
            label = np.argmax(pred.cpu().data.numpy(), axis=1)
            pred_dic[idx - f_range] = int(label)
            del rel_li[0:3]
    while idx <= len(ball_frame) + f_range:
        rel_li += [0,0,0]
        idx += 1
        if len(rel_li) == (f_range * 2 + 1) * 3:
            ball = torch.FloatTensor(rel_li)
            pred = model(ball.unsqueeze(0).to(device))
            label = np.argmax(pred.cpu().data.numpy(), axis=1)
            pred_dic[idx - f_range] = int(label)
            del rel_li[0:3]

    # print(pred_dic)
    return pred_dic

if __name__ == '__main__':
    all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    all_ball = pd.read_csv(f"./csv/all_ball_pos_V2.csv")
    vid = 1
    model = build_model()
    
    get_model_pred(model , all_ball[all_ball['VideoName'] == vid])