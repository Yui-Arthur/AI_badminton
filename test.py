import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import math
# cv2.getPerspectiveTransform(, new)
# vid_id = "00033"
# vid_path = Path(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
# getPerspectiveTransformMatrix(str(vid_path))
from utils.homography_transformation import getPerspectiveTransformMatrix
if __name__ == '__main__':
    pass
    for i in range(1,801): 
        vid_id = str(i).rjust(5,'0')
        print(vid_id)
        vid_path = Path(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
        # matrix , border , _ = getPerspectiveTransformMatrix(str(vid_path))
        matrix , corner , f = getPerspectiveTransformMatrix(str(vid_path) , show_frame=True)
        # if f != -1 :
        #     acc+=1


        # cap = cv2.VideoCapture(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
        # ball_labels = pd.read_csv(f"./ball_pred/ball_pred/{vid_id}_predict.csv")
        # labels = pd.read_csv(f'./part1/part1/train/{vid_id}/{vid_id}_S2.csv')
        # idx = 0
        # show = cv2.warpPerspective(border, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        # while cap.isOpened():

        #     ret, frame = cap.read()

        #     if not ret:
        #         break
            
        #     if ball_labels.loc[ball_labels['Frame'] == idx]['Visibility'].values == 1:

        #         cur_ball_frame = ball_labels.loc[ ball_labels['Frame'] == idx][['X','Y']].values[0]
        #         cv2.circle(frame, cur_ball_frame, 5, (0,255,255), -1)

        #     imgOutput = cv2.warpPerspective(frame, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        #     cv2.imshow('frame',imgOutput)
        #     # cv2.imshow('tt',frame)

        #     idx +=1

        #     # if idx in labels['HitFrame'].values:
        #     #     cv2.waitKey(500)
        #     cv2.waitKey(100)
