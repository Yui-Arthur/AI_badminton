# from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
import random
from multiprocessing import Process, Pool
from itertools import repeat
import math


def get_labels_and_frame(video_folder , imgs_folder, labels_folder):
    
    check_path = labels_folder / f"{video_folder.name}.csv"
    print(video_folder)

    if check_path.exists():
        return

    
    ###### lables ######
    labels = pd.read_csv(f"{video_folder}/{video_folder.name}_S2.csv")
    append_label = pd.DataFrame(columns=labels.columns)

    # 參數 - 要選幾張連接的打擊圖
    pick_hit_frame = 3
    pick_non_hit_frame = [-10 , 10]
    pick_frame = [i - math.floor(pick_hit_frame / 2) for i in range(pick_hit_frame)] + pick_non_hit_frame
    target_frame_num = pick_hit_frame * 2 *  len(labels['HitFrame'])
    
    for idx , sublabel in labels[['HitFrame' , 'ShotSeq']].iterrows():
        sample_label = labels.loc[labels['HitFrame'] == sublabel['HitFrame']]
        label = pd.DataFrame(columns=labels.columns)
        for idx in range(pick_hit_frame):
            label = pd.concat([label , sample_label]).reset_index(drop=True)
            label.at[idx, 'HitFrame'] = sublabel['HitFrame'] - math.floor(pick_hit_frame / 2 ) + idx
            label.at[idx, 'ShotSeq'] = (sublabel['ShotSeq'] - 1) * pick_hit_frame + idx + 1
        
        append_label = pd.concat([append_label, label])
        

    append_label = append_label.reset_index(drop=True)
    append_label.to_csv(labels_folder / f"{video_folder.name}.csv" , index=False)

    ###### frames ######
    cap = cv2.VideoCapture(f'{video_folder}/{video_folder.name}.mp4')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    capture_frame = [i + j for i in labels['HitFrame'] for j in pick_frame]


    while len(capture_frame) < target_frame_num:
        random_frame = random.randint(0 , video_length)

        if random_frame not in capture_frame:
            capture_frame.append(random_frame)

    capture_frame = sorted(capture_frame)


    idx = 0
    # 參數 - 圖片大小 、 裁切範圍 、 縮放倍率
    img_sizex , img_sizey = 1280 , 720
    crop_xt , crop_yt = 200 , 100
    crop_xd , crop_yd = 200 , 0
    resize_factor = 4
    new_sizex = int(( img_sizex - crop_xt - crop_xd ) / resize_factor )
    new_sizey = int(( img_sizey - crop_yt - crop_yd ) / resize_factor )

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        if idx in capture_frame:

            
            frame = frame[crop_yt : img_sizey-crop_yd , crop_xt : img_sizex - crop_xd]
            frame = cv2.resize(frame, (new_sizex , new_sizey), interpolation=cv2.INTER_AREA)


            # cv2.imshow('frame', frame)
            if idx in append_label['HitFrame'].values:
                cv2.imwrite(str(imgs_folder / f"{video_folder.name}_{idx}_hit.jpg"), frame)
            else:
                cv2.imwrite(str(imgs_folder / f"{video_folder.name}_{idx}_x.jpg"), frame)

        idx += 1
        
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    data_folder = Path('./part1/part1/train/').glob('*')
    train_imgs_folder = Path('./dataset/train/images')
    train_imgs_folder.mkdir(parents=True, exist_ok=True)

    train_labels_folder = Path('./dataset/train/labels')
    train_labels_folder.mkdir(parents=True, exist_ok=True)

    valid_imgs_folder = Path('./dataset/valid/images')
    valid_imgs_folder.mkdir(parents=True, exist_ok=True)

    valid_labels_folder = Path('./dataset/valid/labels')
    valid_labels_folder.mkdir(parents=True, exist_ok=True)


    data_folder_list = [i for i in data_folder]
    imgs_folder_list = [valid_imgs_folder if int(i.name.strip('0')) % 10 == 1 else train_imgs_folder for i in data_folder_list ]
    labels_folder_list = [valid_labels_folder if int(i.name.strip('0')) % 10 == 1 else train_labels_folder for i in data_folder_list ]

    # print(labels_folder_list)
    

    CPU_Core_num = 6
    pool = Pool(processes = CPU_Core_num)
    pool.starmap(get_labels_and_frame, zip(data_folder_list , imgs_folder_list , labels_folder_list) , chunksize = int(len(data_folder_list) / CPU_Core_num))




   


