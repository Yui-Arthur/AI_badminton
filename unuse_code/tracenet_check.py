import cv2

import pandas as pd
import numpy as np

vid_id = "00703"

cap = cv2.VideoCapture(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')

labels = pd.read_csv(f'./part1/part1/train/{vid_id}/{vid_id}_S2.csv')
ball_labels = pd.read_csv(f"./all_ball_pos.csv")
print(ball_labels['VideoName'])
ball_labels = ball_labels[ball_labels['VideoName'] == int(vid_id)]
last_ball_frame = [0,0]
last_idx = 0
idx = 0
plot_circle = [(0,0) for i in range(10)]

ball_graph = np.zeros((720,1280,3), np.uint8)
color = [15, 50 , 60]
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break
    

    if ball_labels.loc[ball_labels['Frame'] == idx]['Visibility'].values == 1:

        cur_ball_frame = ball_labels.loc[ ball_labels['Frame'] == idx][['X','Y']].values[0]

        for i in range(9,0,-1):
            plot_circle[i] = plot_circle[i-1]
        
        plot_circle[0] = cur_ball_frame

        cv2.circle(ball_graph, cur_ball_frame, 5, (color[0] % 255 + 15,color[1] % 255 + 15,color[2] % 255 + 15), 2)
        color[0]+=10
        color[1]+=10
        color[2]+=10
        
        # speed =  ((cur_ball_frame[0] - last_ball_frame[0])**2 + (cur_ball_frame[1] - last_ball_frame[1])**2) / ( idx - last_idx )  
        # last_idx = idx
        # last_ball_frame = cur_ball_frame
        # print(speed)
        # if speed > 1000:
        #     cv2.waitKey(500)
        # print(labels.loc[ labels['Frame'] == idx][['X','Y']].values[0])

    for i in plot_circle:
        print(i)
        cv2.circle(frame, i, 5, (0,0,255), 2)

    cv2.imshow('frame',ball_graph)

    if idx in labels['HitFrame'].values:
        cv2.waitKey(10)
    # print(labels.iloc[idx]['Visibility'])
    # for i in range(prev_idx-1 , 0 , -1):
    #     prev_frame[i] = prev_frame[i-1]

    # prev_frame[0] = frame
    cv2.waitKey(10)
    idx+=1
    # while cv2.waitKey(50) == -1:
    #     pass

