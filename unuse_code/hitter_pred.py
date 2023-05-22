import pandas as pd
import numpy as np
import cv2




def show_frame(vid , odd_hit_range , even_hit_range):
    cap = cv2.VideoCapture(f"./data/part1/train/{str(vid).rjust(5,'0')}/{str(vid).rjust(5,'0')}.mp4")
    idx = 1
    blank_image = np.zeros((720,1280,3), np.uint8)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx in even_hit_range and not ball_labels[ball_labels['Frame'] == idx].empty:
            cv2.circle(blank_image , ball_labels[ball_labels['Frame'] == idx][['X','Y']].values[0] , 5 , (0,0,255) , -1)
        elif idx in odd_hit_range and not ball_labels[ball_labels['Frame'] == idx].empty:
            cv2.circle(blank_image , ball_labels[ball_labels['Frame'] == idx][['X','Y']].values[0] , 5 , (0,255,0) , -1)

        img2gray = cv2.cvtColor(blank_image,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(frame,frame,mask = mask_inv)
        img2_fg = cv2.bitwise_and(blank_image,blank_image,mask = mask)
        frame = cv2.add(img1_bg,img2_fg)

        cv2.imshow('f',frame)
        while cv2.waitKey(10) == -1:
            pass
            break
        idx += 1

    while cv2.waitKey(10) == -1:
        pass


def guess_hitter(ball_labels , hit_labels):
    

    ball_labels = ball_labels.drop(ball_labels[ball_labels['Visibility'] == 0].index)


    odd_labels = hit_labels[(hit_labels['ShotSeq'] % 2 == 1)]['HitFrame']
    even_labels = hit_labels[(hit_labels['ShotSeq'] % 2 == 0)]['HitFrame']
    

    frame_range = 3
    odd_hit_range = [i+j for i in odd_labels.values for j in range(-frame_range,frame_range+1)]
    even_hit_range = [i+j for i in even_labels.values for j in range(-frame_range,frame_range+1)]
    
    odd_ball_labels = ball_labels[(ball_labels['Frame'].isin(odd_hit_range))]['Y'].values
    even_ball_labels = ball_labels[(ball_labels['Frame'].isin(even_hit_range))]['Y'].values
                                #    


    y_range = 0
    if((odd_ball_labels.mean() - y_range  > even_ball_labels.mean())):
        return "B"
    elif ((odd_ball_labels.mean() - y_range  < even_ball_labels.mean())):
        return "A"
    else:
        return -1
        

all_hit_labels = pd.read_csv(f"./csv/all_hit_labels.csv")
all_ball_labels = pd.read_csv(f"./csv/all_ball_pos_V2.csv")
success = 0
fail = []

for vid in range(1,801):
    # vid = str(vid).rjust(5,"0")
    # print(vid)
    hit_labels = all_hit_labels[all_hit_labels['VideoName'] == vid]
    ball_labels = all_ball_labels[all_ball_labels['VideoName'] == vid]
    
    guess = guess_hitter(ball_labels , hit_labels)

    if guess == hit_labels.iloc[0]['Hitter']:
        success += 1
        
    # show_frame(vid , odd_hit_range ,even_hit_range)



print(success/800)
print(fail)