import numpy as np
import cv2
import pandas as pd
from pathlib import Path


width , height = 1280 , 720

def getPerspectiveTransformMatrix(vid_path):

    cap = cv2.VideoCapture(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
    # width , height = 1080 , 720
    border = np.zeros((720,1280,3), np.uint8)
    test_frame = None
    print("Start")
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break
        test_frame = frame

        # 灰階
        imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 二值化
        ret,thresh = cv2.threshold(imgray,190,255,0)
        # 找輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = list(contours)
        # 根據輪廓面積大小進行sort
        contours.sort(key = cv2.contourArea , reverse=True)
        # 畫出前20的輪廓
        cv2.drawContours(border, contours[0:20], -1, (0,0,255), 10)

        # 每幀全部 -3
        cv2.subtract(border,(3,3,3,0) , border)

        cv2.imshow('frame' , border)
        while cv2.waitKey(1) == -1:
            pass
            break

    # 球場轉灰階
    imgray = cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)
    # 變黑白
    ret,thresh = cv2.threshold(imgray,50,255,0)
    # 找邊緣
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 找最大的面積 = 球場
    c = max(contours, key = cv2.contourArea)
    # 找凸包
    hull = cv2.convexHull(c)
    epsilon = 0.01*cv2.arcLength(hull,True)
    approx = cv2.approxPolyDP(hull,epsilon,True)
    # 劃出球場
    clean_border = np.zeros((720,1280,3), np.uint8)

    cv2.drawContours(clean_border, [approx], -1, (0,255,255), 2)

    cv2.imshow('test', clean_border)

    # 進行透視變換
    old = np.float32(approx)
    new = np.float32([[0,height-1] , [0,0], [width-1,0], [width-1,height-1]])
    matrix = cv2.getPerspectiveTransform(old , new)
    imgOutput = cv2.warpPerspective(test_frame, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    
    cv2.imshow('tt', imgOutput)
    while cv2.waitKey(10) == -1:
        pass
    cv2.destroyAllWindows()
    


    return matrix , clean_border , approx


# cv2.getPerspectiveTransform(, new)
vid_id = "00180"
vid_path = Path(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
 
matrix , border , _ = getPerspectiveTransformMatrix(str(vid_path))

# cap = cv2.VideoCapture(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
# ball_labels = pd.read_csv(f"./{vid_id}_predict.csv")
# labels = pd.read_csv(f'./part1/part1/train/{vid_id}/{vid_id}_S2.csv')
# idx = 0
# show = cv2.warpPerspective(border, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
# while cap.isOpened():

#     ret, frame = cap.read()

#     if not ret:
#         break
    
#     if ball_labels.loc[ball_labels['Frame'] == idx]['Visibility'].values == 1:

#         cur_ball_frame = ball_labels.loc[ ball_labels['Frame'] == idx][['X','Y']].values[0]
#         # show = np.copy(border)
#         # cv2.circle(show, cur_ball_frame, 5, (0,255,255), -1)
#         # imgOutput = cv2.warpPerspective(show, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#         cv2.circle(frame, cur_ball_frame, 5, (0,255,255), -1)

#     imgOutput = cv2.warpPerspective(frame, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
#     cv2.imshow('frame',imgOutput)
#     # cv2.imshow('tt',frame)

#     idx +=1

#     # if idx in labels['HitFrame'].values:
#     #     cv2.waitKey(500)
#     cv2.waitKey(10)