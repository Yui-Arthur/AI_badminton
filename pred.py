from utils.get_hit_frame import get_hit_pred 
from utils.get_hitter import guess_hitter
import pandas as pd
import numpy as np


def get_confusion_matrix():
    all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    confusion_matrix = np.zeros((2, 2) , dtype=np.int64)
    success_vid = 0
    for vid in range(1,801):
        # print(vid)
        ball_csv = f"./data/ball_pred_V2/{str(vid).rjust(5,'0')}_ball.csv"
        try:
            pred = get_hit_pred(ball_csv)
        except:
            pred = []
        hit_labels = all_hit_labels[all_hit_labels['VideoName'] == vid]['HitFrame'].values

        success_frame = []
        for pred_frame in pred:
            pred_ac = False
            for pred_range in range(-2,3):
                if pred_frame + pred_range in hit_labels:
                    pred_ac = True
                    success_frame.append(pred_frame + pred_range)
                    break
            
            if pred_ac:
                confusion_matrix[0][0] +=1
            else:
                confusion_matrix[1][0] +=1

        if len(success_frame) == len(hit_labels):
            success_vid +=1

        confusion_matrix[0][1] += len(hit_labels) - len(success_frame)

    print(confusion_matrix)
    print(success_vid)
    print(np.diag(confusion_matrix)/confusion_matrix.sum(1))


result = pd.read_csv("./csv/sample_result.csv")
all_ball_labels = pd.read_csv(f"./csv/all_ball_pos_V2.csv")

for vid in range(1,170):
        # print(vid)
    ball_csv = f"./data/ball_pred_V2/{str(vid).rjust(5,'0')}_ball.csv"
    try:
        pred = get_hit_pred(ball_csv)
    except:
        pred = []
    
    tmp = result

    for idx , frame in enumerate(pred):
        _ = pd.DataFrame([[str(vid).rjust(5,'0') , idx+1 , frame , "A"  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, 'X']], columns=result.columns)
        tmp = pd.concat((tmp , _))

    hit_labels = result[result['VideoName'] == vid]
    ball_labels = all_ball_labels[all_ball_labels['VideoName'] == vid]
    # print(result)
    hitter = guess_hitter(ball_labels , tmp) 
    if hitter == -1:
        hitter = 'A'
    win = 'X'
    for idx , frame in enumerate(pred):

        if idx + 1 == len(pred):
            win = 'A'

        _ = pd.DataFrame([[str(vid).rjust(5,'0')+".mp4" , idx+1 , frame , hitter  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, win]], columns=result.columns)
        result = pd.concat((result , _))
        hitter = "B" if hitter == 'A' else 'A'




    
result.to_csv("./csv/0513_result.csv" , index=False)
        
