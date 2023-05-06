import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import math

width , height = 1280 , 720
def get_angle(line1, line2):
    # Get directional vectors
    d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    # Compute dot product
    p = d1[0] * d2[0] + d1[1] * d2[1]
    # Compute norms
    n1 = math.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = math.sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    # Compute angle
    ang = math.acos(p / (n1 * n2))
    # Convert to degrees if you want
    ang = math.degrees(ang)
    return ang
def getPerspectiveTransformMatrix(vid_path):

    cap = cv2.VideoCapture(vid_path)
    # width , height = 1080 , 720
    border = np.zeros((720,1280,3), np.uint8)
    test_frame = None
    print("Start")
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break
        test_frame = frame

        h, w = frame.shape[:2]
        mask = np.zeros([h+2, w+2], np.uint8) 
        diff_value = (5,5,5)
        black = (0,0,0)
        frame = cv2.GaussianBlur(frame , (5,5) , 0)
        cv2.floodFill(frame, mask, (100,0), black, diff_value, diff_value)
        cv2.floodFill(frame, mask, (100,700), black, diff_value, diff_value)
        cv2.floodFill(frame, mask, (1100,700), black, diff_value, diff_value)
        cv2.circle(frame , (100,0) , 5 ,(255,0,) , -1)
        cv2.circle(frame , (1100,700) , 5 ,(255,0,) , -1)
        # print(frame[500,100])
        # 灰階
        imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 二值化
        ret,thresh = cv2.threshold(imgray,100,255,0)
        # 找輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = list(contours)
        # 根據輪廓面積大小進行sort
        contours.sort(key = cv2.contourArea , reverse=True)
        # 畫出前20的輪廓
        cv2.drawContours(border, contours[0:1], -1, (0,0,255), 10)

        # 每幀全部 -3
        cv2.subtract(border,(3,3,3,0) , border)

        # 灰階
        imgray = cv2.cvtColor(border,cv2.COLOR_BGR2GRAY)
        # 二值化
        # ret,thresh = cv2.threshold(imgray,50,255,0)

        canny = cv2.Canny(imgray, 30, 100)
        lines = cv2.HoughLinesP(canny , 1.0 , np.pi/180 , 100 , np.array([]) , 200 , 100)
        # lines_edges = cv2.addWeighted(frame, 0.8, line, 1, 0)
        # print(lines[0])
        cv2.line(frame,(100,100),(100,500),(255,255,255),5)
        try:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    line_angle = get_angle([(x1,y1),(x2,y2)] , [(0,0),(0,100)])
                    line_angle = 180 - line_angle if line_angle > 90 else line_angle
                    
                    if  line_angle < 40 or line_angle > 85: 
                        
                        vectorx = x2 - x1
                        vectory = y2 - y1
                        print(vectorx , vectory)
                        cv2.line(frame,(x1 - vectorx * 100, y1 - vectory * 100),(x1 + vectorx * 100,y1 + vectory * 100),(255,0,0),5)
        except:
            return
        # cv2.imshow('thresh' , thresh)     
        # cv2.imshow('board' , border)          
        cv2.imshow('frame' , frame)

        while cv2.waitKey(100) == -1:
            pass
            break
        return
            

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
vid_id = "00017"
for i in range(1,801): 
    vid_id = str(i).rjust(5,'0')
    print(vid_id)
    vid_path = Path(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
    # matrix , border , _ = getPerspectiveTransformMatrix(str(vid_path))
    getPerspectiveTransformMatrix(str(vid_path))

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