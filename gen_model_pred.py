from utils.get_hit_frame import get_hit_pred 
from utils.get_hitter import guess_hitter
from utils.get_model_pred import build_model , get_model_pred
# from utils.get_model_pred_with_frame import build_model , get_model_pred
import pandas as pd
import numpy as np
import json

all_ball = pd.read_csv(f"./csv/all_ball_pos_test_V2_smooth.csv")
model = build_model()
total_pred = {}

for vid in range(170,400):
    print(vid)

    # model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid] , f"./data/part1/train/{str(vid).rjust(5,'0')}/{str(vid).rjust(5,'0')}.mp4")
    model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid])
    total_pred[vid] = model_pred
    # break
with open('./data/test_pred_ball_smooth.json' , 'w') as f:
    json.dump(total_pred , f)