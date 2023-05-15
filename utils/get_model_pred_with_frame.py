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
from utils.homography_transformation import getPerspectiveTransformMatrix
import cv2

class Classifier(nn.Module):
    def __init__(self , nb_classes):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.cnn_fc = nn.Linear(1000 , nb_classes)
        
        self.ball_prefc = nn.LazyLinear(100)
        self.ball_attention = nn.TransformerEncoderLayer(100 , 1 , dim_feedforward = 128 , dropout=0.15 , batch_first = True)
        self.ball_fc = nn.LazyLinear(nb_classes)
        self.fin_fc = nn.LazyLinear(nb_classes)
        
    def forward(self, img , ball):
        
        imgout = self.cnn(img)
        imgout = self.cnn_fc(imgout)
        
        ballout = self.ball_prefc(ball)
        ballout = self.ball_attention(ballout)
        ballout = self.ball_fc(ballout)
        out = self.fin_fc(torch.cat((imgout,ballout) , dim = 1))
        
        
        return out
    


def build_model():    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nb_classes = 2
    model = Classifier(nb_classes).to(device)
    _exp_name = "hit_model"
    read_model = "./models/hit_cnn_ball_model_bestckpt.ckpt"

    model.load_state_dict(torch.load(read_model))

    model.eval()
    return model

def get_model_pred(model , ball_frame , vid_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f_range = 30

    rel_li = []
    pred_dic = {}
    
    idx = -f_range 
    # for _ in range(-f_range + 1 , len(ball_frame) + f_range):

    cap = cv2.VideoCapture(vid_path)
    matrix , corner , f = getPerspectiveTransformMatrix(vid_path)
    if f == -1:
        print("Corner Failed")
        return []
        
    corner[0][1] -= 70
    corner[3][1] -= 70
    new_sizex , new_sizey = 330,150
    img_sizex , img_sizey = 1280 , 720

    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])

    for _ , j in ball_frame.iterrows():
        # print(idx)

        while idx < j['Frame']:
            rel_li += [0,0,0]
            idx += 1

        if idx == j['Frame']: 
            rel_li += [int(j['Visibility']),int(j['X']),int(j['Y'])]
            idx += 1

        if len(rel_li) == (f_range * 2 + 1) * 3:
            ret , frame = cap.read()
            imgOutput = cv2.warpPerspective(frame, matrix, (img_sizex , img_sizey), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            imgOutput = cv2.resize(imgOutput, (new_sizex , new_sizey), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))
            img = tfm(img)
            ball = torch.FloatTensor(rel_li)
            # print(ball)
            # print(img.shape)
            pred = model(img.unsqueeze(0).to(device) , ball.unsqueeze(0).to(device))
            label = np.argmax(pred.cpu().data.numpy(), axis=1)
            pred_dic[idx - f_range] = int(label)
            del rel_li[0:3]
    while idx <= len(ball_frame) + f_range:
        rel_li += [0,0,0]
        idx += 1
        if len(rel_li) == (f_range * 2 + 1) * 3:
            ball = torch.FloatTensor(rel_li)
            ret , frame = cap.read()
            if not ret:
                break
            imgOutput = cv2.warpPerspective(frame, matrix, (img_sizex , img_sizey), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            imgOutput = cv2.resize(imgOutput, (new_sizex , new_sizey), interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))
            img = tfm(img)
            pred = model(img.unsqueeze(0).to(device) , ball.unsqueeze(0).to(device))
            label = np.argmax(pred.cpu().data.numpy(), axis=1)
            pred_dic[idx - f_range] = int(label)
            del rel_li[0:3]

    # print(pred_dic)
    return pred_dic

if __name__ == '__main__':
    pass
    # all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    # all_ball = pd.read_csv(f"./csv/all_ball_pos_V2.csv")
    # vid = 1
    # model = build_model()
    
    # get_model_pred(model , all_ball[all_ball['VideoName'] == vid])