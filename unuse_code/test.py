import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import math
from utils.homography_transformation import getPerspectiveTransformMatrix


if __name__ == '__main__':
    width , height = 1280 , 720
    
    for i in range(1,801): 
        vid_id = str(i).rjust(5,'0')
        print(vid_id)
        vid_path = Path(f'./data/part1/train/{vid_id}/{vid_id}.mp4')
        matrix , border , _ = getPerspectiveTransformMatrix(str(vid_path) , show_frame=True)
        # matrix , corner , f = getPerspectiveTransformMatrix(str(vid_path))

        # corner[0][1] -= 70
        # corner[3][1] -= 70
        

        # old = np.float32(corner)
        # new = np.float32([[0,0], [0,height-1], [width-1,height-1] , [width-1,0]])
        # matrix = cv2.getPerspectiveTransform(old , new)
        

        # cap = cv2.VideoCapture(f'./data/part1/train/{vid_id}/{vid_id}.mp4')
        # ball_labels = pd.read_csv(f"./ball_pred/ball_pred/{vid_id}_predict.csv")
        # labels = pd.read_csv(f'./part1/part1/train/{vid_id}/{vid_id}_S2.csv')
        # idx = 0

        # while cap.isOpened():

        #     ret, frame = cap.read()

        #     if not ret:
        #         break
        #     # print("!")
        #     frame_num += 1
        #     # if ball_labels.loc[ball_labels['Frame'] == idx]['Visibility'].values == 1:

            #     cur_ball_frame = ball_labels.loc[ ball_labels['Frame'] == idx][['X','Y']].values[0]
            #     cv2.circle(frame, cur_ball_frame, 5, (0,255,255), -1)

            # imgOutput = cv2.warpPerspective(frame, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            # cv2.imshow('frame',imgOutput)
            # # cv2.imshow('tt',frame)

            # idx +=1

            # if idx in labels['HitFrame'].values:
            #     cv2.waitKey(500)
            # cv2.waitKey(10)
