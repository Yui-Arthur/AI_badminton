import cv2

import pandas as pd

vid_id = "00167"

cap = cv2.VideoCapture(f'./part1/part1/val/{vid_id}/{vid_id}.mp4')

labels = pd.read_csv(f"./result.csv")


idx = 1
prev_idx = 0
prev_frame = [0 for i in range(prev_idx)]

frame = []

labels = labels[labels['VideoName'] == f"{vid_id}.mp4"]

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # diff = cv2.subtract(prev_frame[-1] , frame)
    if idx in labels['HitFrame'].add(prev_idx).values and idx > prev_idx:
        # print(frame , prev_frame[-1])
        # diff = cv2.subtract(frame, prev_frame[-1])
        # cv2.imshow('frame', diff)
        # cv2.rectangle(diff, (20, 60), (120, 160), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        while cv2.waitKey(1000) == -1:
            pass

    # elif idx in labels['HitFrame'].values and idx > prev_idx:
        # cv2.imshow('frame', diff)
    elif idx > prev_idx:
        # cv2.imshow('frame', diff)
        # while(cv2.waitKey() == -1):
        pass
    # if cv2.waitKey(100) == ord('q'):
    #         break
    idx += 1
    

    # for i in range(prev_idx-1 , 0 , -1):
    #     prev_frame[i] = prev_frame[i-1]

    # prev_frame[0] = frame

