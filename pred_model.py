from utils.get_hit_frame import get_hit_pred 
from utils.get_hitter import guess_hitter
# from utils.get_model_pred import build_model , get_model_pred
# from utils.get_model_pred_with_frame import build_model , get_model_pred
import pandas as pd
import numpy as np
import json


def get_confusion_matrix():
    all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
    # smooth = "_smooth"
    smooth = ""
    confusion_matrix = np.zeros((2, 2) , dtype=np.int64)
    success_vid = 0
    # model = build_model()
    with open(f"./model_pred/train_pred_ball_with_img.json") as f:
        all_model_pred_with_img = json.load(f)
    with open(f"./model_pred/train_pred_ball_smooth_with_img.json") as f:
        all_model_pred_smooth_with_img = json.load(f)
    with open(f"./model_pred/train_pred_ball.json") as f:
        all_model_pred = json.load(f)
    with open(f"./model_pred/train_pred_ball_smooth.json") as f:
        all_model_pred_smooth = json.load(f)
    # print(all_model_pred)
    del_num = 0
    for vid in range(1,801):
        # print(vid)
        ball_csv = f"./data/ball_pred_V2{smooth}/{str(vid).rjust(5,'0')}_ball.csv"
        try:
            pred = get_hit_pred(ball_csv)
        except:
            pred = []
        # print(pred)
        # model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid] , f"./data/part1/train/{str(vid).rjust(5,'0')}/{str(vid).rjust(5,'0')}.mp4")
        # model_pred = get_model_pred(model , all_ball[all_ball['VideoName'] == vid]")
        model_pred_with_img = all_model_pred_with_img[str(vid)]
        model_pred = all_model_pred[str(vid)]
        model_pred_smooth_with_img = all_model_pred_smooth_with_img[str(vid)]
        model_pred_smooth = all_model_pred_smooth[str(vid)]
        # print(model_pred)
        # print(pred)
        del_frame = []
        for frame in pred:
            pred_hit = 0
            
            for _ in range(-7,8):
                tt =  str(frame - _)
                if tt not in model_pred or tt not in model_pred_with_img:
                    continue

                if model_pred[tt] == 0 or model_pred_with_img[tt] == 0 or model_pred_smooth_with_img[tt] == 0 or model_pred_smooth[tt] == 0:
                    pred_hit +=1
               

            if pred_hit < 1:
                del_frame.append(frame)

    

        for f in del_frame:
            pred.remove(f)
            del_num += 1
            pass
                
        
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
    print("all hit num video : " , success_vid)
    # print(np.diag(confusion_matrix)/confusion_matrix.sum(1))
    print(del_num)
    print((169 * (success_vid / 800) *0.17) / 169)

# get_confusion_matrix()
# exit()

result = pd.read_csv("./csv/sample_result.csv")
smooth = ""
state = "test"
all_ball = pd.read_csv(f"./csv/all_ball_pos_{state}_V2{smooth}.csv")
with open(f"./model_pred/train_pred_ball_with_img.json") as f:
        all_model_pred_with_img = json.load(f)
with open(f"./model_pred/train_pred_ball_smooth_with_img.json") as f:
    all_model_pred_smooth_with_img = json.load(f)
with open(f"./model_pred/train_pred_ball.json") as f:
    all_model_pred = json.load(f)
with open(f"./model_pred/train_pred_ball_smooth.json") as f:
    all_model_pred_smooth = json.load(f)

for vid in range(170,400):
        # print(vid)
    ball_csv = f"./data/ball_pred_{state}_V2_smooth/{str(vid).rjust(5,'0')}_ball.csv"
    try:
        pred = get_hit_pred(ball_csv)
    except:
        pred = []
    
    
    model_pred_with_img = all_model_pred_with_img[str(vid)]
    model_pred = all_model_pred[str(vid)]
    model_pred_smooth_with_img = all_model_pred_smooth_with_img[str(vid)]
    model_pred_smooth = all_model_pred_smooth[str(vid)]
    # print(pred)
    del_frame = []
    for frame in pred:
        pred_hit = 0
        
        for _ in range(-7,8):
            tt =  str(frame - _)
            if tt not in model_pred or tt not in model_pred_with_img:
                continue

            if model_pred[tt] == 0 or model_pred_with_img[tt] == 0 or model_pred_smooth_with_img[tt] == 0 or model_pred_smooth[tt] == 0:
                pred_hit +=1
            

        if pred_hit < 1:
            del_frame.append(frame)



    for f in del_frame:
        pred.remove(f)
        # del_num += 1
        pass


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




    
result.to_csv("./csv/0516_1_result.csv" , index=False)
        
