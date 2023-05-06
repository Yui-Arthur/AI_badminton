import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import math


width , height = 1280 , 720
def get_angle(line1, line2):
    d1 = (line1[1][0] - line1[0][0], line1[1][1] - line1[0][1])
    d2 = (line2[1][0] - line2[0][0], line2[1][1] - line2[0][1])
    p = d1[0] * d2[0] + d1[1] * d2[1]
    n1 = math.sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    n2 = math.sqrt(d2[0] * d2[0] + d2[1] * d2[1])
    ang = math.acos(p / (n1 * n2))
    ang = math.degrees(ang)
    return ang
def get_intersections(line1, line2):
    A = np.array(line1)
    B = np.array(line2)
    t, s = np.linalg.solve(np.array([A[1]-A[0], B[0]-B[1]]).T, B[0]-A[0])    
    
    return (1-t)*A[0] + t*A[1]
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
        # bgr
        diff_value = (3,1,3)
        black = (0,0,0)
        frame = cv2.GaussianBlur(frame , (7,7) , 0)

        fillpoints = [(100,0) , (100,700) , (300,350) , (1000,350) , (1200,700)]
        
        for point in fillpoints:
            cv2.floodFill(frame, mask, point, black, diff_value, diff_value)
        for point in fillpoints:
            cv2.circle(frame , point , 5 ,(0,0,255) , -1)

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

        horizon_line = []
        left_vertical_line = None
        right_vertical_line = None

        try:
        # 畫球場的垂直線 和 找到水平線座標
            for line in lines:
                for x1,y1,x2,y2 in line:
                    line_angle = get_angle([(x1,y1),(x2,y2)] , [(0,0),(0,100)])
                    line_angle_90 = 180 - line_angle if line_angle > 90 else line_angle
                    vectorx = x2 - x1
                    vectory = y2 - y1
                    
                    if  (line_angle_90 < 40 and int(line_angle)): 
                        
                        

                        if left_vertical_line == None and line_angle > 90:
                            # left_vertical_line = [(x1 - 20 , y1) , (x2 -20, y2)]
                            left_vertical_line = [(x1 , y1) , (x2, y2)]
                            cv2.line(frame,(x1 - vectorx * 100, y1 - vectory * 100),(x1 + vectorx * 100,y1 + vectory * 100),(255,0,0),5)
                        elif right_vertical_line == None and line_angle < 90:
                            right_vertical_line = [(x1, y1) , (x2, y2)]
                            cv2.line(frame,(x1 - vectorx * 100, y1 - vectory * 100),(x1 + vectorx * 100,y1 + vectory * 100),(255,0,0),5)

                    elif line_angle_90 > 85:

                        horizon_line.append([[x1 ,y1] , [x2,y2] ])

            # 畫上下兩條水平線
            top_line = min(horizon_line , key = lambda x : x[0][1] + x[1][1])
            # top_line[0][1] -= 20
            # top_line[1][1] -= 20
            x1 , y1 = top_line[0]
            x2 , y2 = top_line[1]
            cv2.line(frame,(x1 - (x2-x1) * 100, y1 - (y2-y1) * 100),(x1 + (x2-x1) * 100,y1 + (y2-y1) * 100),(255,0,0),5)    
            bottom_line = max(horizon_line , key = lambda x : x[0][1] + x[1][1])
            print(bottom_line)
            x1 , y1 = bottom_line[0]
            x2 , y2 = bottom_line[1]
            cv2.line(frame,(x1 - (x2-x1) * 100, y1 - (y2-y1) * 100),(x1 + (x2-x1) * 100,y1 + (y2-y1) * 100),(255,0,0),5)    

            # print(get_intersections(top_line , vertical_line[0]).astype(int))
            corner = []
            corner.append(get_intersections(top_line , left_vertical_line).astype(int))
            corner.append(get_intersections(bottom_line , left_vertical_line).astype(int))
            corner.append(get_intersections(bottom_line , right_vertical_line).astype(int))
            corner.append(get_intersections(top_line , right_vertical_line).astype(int))
            cv2.circle(frame , get_intersections(top_line , left_vertical_line).astype(int) , 5 , (0,255,0),-1)
            cv2.circle(frame , get_intersections(top_line , right_vertical_line).astype(int) , 5 , (0,255,0),-1)
            cv2.circle(frame , get_intersections(bottom_line , left_vertical_line).astype(int) , 5 , (0,255,0),-1)
            cv2.circle(frame , get_intersections(bottom_line , right_vertical_line).astype(int) , 5 , (0,255,0),-1)
            
        except:
            return -1 , -1
            

        # cv2.imshow('board' , border)          
        cv2.imshow('frame' , frame)
        while cv2.waitKey(1) == -1:
            pass
            break
        #     if vertical_line_cnt == 2:
        #         break
        #     # cv2.imshow('thresh' , thresh)     
        #     cv2.imshow('board' , border)  
        #     cv2.imshow('frame' , frame)

        break
            
    # 進行透視變換
    old = np.float32(corner)
    new = np.float32([[0,0], [0,height-1], [width-1,height-1] , [width-1,0] ])
    matrix = cv2.getPerspectiveTransform(old , new)
    imgOutput = cv2.warpPerspective(test_frame, matrix, (width , height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    
    # cv2.imshow('tt', imgOutput)
    while cv2.waitKey(200) == -1:
        pass
        break
    # cv2.destroyAllWindows()
    


    return matrix , corner


# cv2.getPerspectiveTransform(, new)
# vid_id = "00033"
# vid_path = Path(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
# getPerspectiveTransformMatrix(str(vid_path))
global acc
acc = 0

for i in range(1,801): 
    vid_id = str(i).rjust(5,'0')
    print(vid_id)
    vid_path = Path(f'./part1/part1/train/{vid_id}/{vid_id}.mp4')
    # matrix , border , _ = getPerspectiveTransformMatrix(str(vid_path))
    matrix , corner = getPerspectiveTransformMatrix(str(vid_path))

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
