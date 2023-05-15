from utils.get_hit_frame import get_hit_pred 
from utils.get_hitter import guess_hitter
# from utils.get_model_pred import build_model , get_model_pred
from utils.get_model_pred_with_frame import build_model , get_model_pred
import pandas as pd
import numpy as np
import json


def get_confusion_matrix():
    all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    all_ball = pd.read_csv(f"./csv/all_ball_pos_V2_smooth.csv")
    confusion_matrix = np.zeros((2, 2) , dtype=np.int64)
    success_vid = 0
    # model = build_model()
    with open(f"./data/train_pred_ball_smooth_with_img.json") as f:
        all_model_pred = json.load(f)
    # print(all_model_pred)
    for vid in range(1,801):
        # print(vid)
        ball_csv = f"./data/ball_pred_V2_smooth/{str(vid).rjust(5,'0')}_ball.csv"
        try:
            pred = get_hit_pred(ball_csv)
        except:
            pred = []
        # print(pred)
        # model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid] , f"./data/part1/train/{str(vid).rjust(5,'0')}/{str(vid).rjust(5,'0')}.mp4")
        # model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid]")
        model_pred = all_model_pred[str(vid)]
        # print(model_pred)
        # print(pred)
        del_frame = []
        for frame in pred:
            pred_hit = 0
            
            for _ in range(-7,8):
                if str(frame - _)  in model_pred and model_pred[str(frame - _)] == 0:
                    pred_hit +=1

            if pred_hit < 1:
                del_frame.append(frame)
        # print(del_frame)
        # print("del ",len(del_frame))
        for f in del_frame:
            pred.remove(f)
            # pass
                
        
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

        if len(pred) == len(hit_labels):
            success_vid +=1

        confusion_matrix[0][1] += len(hit_labels) - len(success_frame)
        # break
    print(confusion_matrix)
    print(success_vid)
    print(np.diag(confusion_matrix)/confusion_matrix.sum(1))
    print((169 * (success_vid / 800) *0.17) / 169)

get_confusion_matrix()
exit()

result = pd.read_csv("./csv/sample_result.csv")
all_ball = pd.read_csv(f"./csv/all_ball_pos_valid_V2_smooth.csv")
model = build_model()
for vid in range(1,170):
        # print(vid)
    ball_csv = f"./data/ball_pred_valid_V2_smooth/{str(vid).rjust(5,'0')}_ball.csv"
    try:
        pred = get_hit_pred(ball_csv)
    except:
        pred = []
    
    model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid])
    del_frame = []
    for frame in pred:
        pred_hit = 0
        for _ in range(-5,6):
            if frame - _  in model_pred and model_pred[frame - _] == 0:
                pred_hit +=1
        if pred_hit < 1:
            del_frame.append(frame)
    
    # print("del ",len(del_frame))
    for f in del_frame:
        pred.remove(f)


    tmp = result

    for idx , frame in enumerate(pred):
        _ = pd.DataFrame([[str(vid).rjust(5,'0') , idx+1 , frame , "A"  , 1 , 1 , 1 , 1 ,1 ,1 ,1 ,1 ,1 ,1, 'X']], columns=result.columns)
        tmp = pd.concat((tmp , _))

    hit_labels = result[result['VideoName'] == vid]
    ball_labels = all_ball[all_ball['VideoName'] == vid]
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




    
result.to_csv("./csv/0514_result.csv" , index=False)
        
