# from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
import random
from multiprocessing import Process, Pool
from itertools import repeat
import math
from argparse import Namespace
from utils.homography_transformation import getPerspectiveTransformMatrix
import numpy as np


def hit_label_append(video_folder , labels_folder , config):
    ##### hit lables ######
    labels = pd.read_csv(f"{video_folder}/{video_folder.name}_S2.csv")
    append_label = pd.DataFrame(columns=labels.columns)

    # 參數 - 要選幾張連接的打擊圖
    pick_hit_frame = config.pick_hit_frame
    pick_non_hit_frame = config.pick_non_hit_frame
    random_size = len(pick_non_hit_frame) if str(labels_folder).split('\\')[1] == 'train' else len(pick_non_hit_frame) * config.valid_random_factor
    pick_frame = [i - math.floor(pick_hit_frame / 2) for i in range(pick_hit_frame)] + pick_non_hit_frame
    
    # only one hitter
    # labels.drop(labels[labels['Hitter'] == 'A'].index , inplace = True)
    target_frame_num = (pick_hit_frame + len(pick_non_hit_frame) + random_size) *  len(labels['HitFrame'])
    # print(labels['Hitter'])
    # hit label apppend
    for idx , sublabel in labels[['HitFrame' , 'ShotSeq']].iterrows():
        sample_label = labels.loc[labels['HitFrame'] == sublabel['HitFrame']]
        label = pd.DataFrame(columns=labels.columns)
        for idx in range(pick_hit_frame):
            label = pd.concat([label , sample_label]).reset_index(drop=True)
            label.at[idx, 'HitFrame'] = sublabel['HitFrame'] - math.floor(pick_hit_frame / 2 ) + idx
            label.at[idx, 'ShotSeq'] = (sublabel['ShotSeq'] - 1) * pick_hit_frame + idx + 1
        
        append_label = pd.concat([append_label, label])
        
    append_label = append_label.reset_index(drop=True)
    # new appended label csv
    append_label.to_csv(labels_folder / f"{video_folder.name}.csv" , index=False)

    
    capture_frame = {i + j:0 if j == 0 else 1 for i in labels['HitFrame'] for j in pick_frame}

    return target_frame_num , capture_frame   

def get_frame(video_folder , imgs_folder , capture_frame , target_frame_num , config):
    cap = cv2.VideoCapture(f'{video_folder}/{video_folder.name}.mp4')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while len(capture_frame) < target_frame_num and len(capture_frame) != video_length:
        random_frame = random.randint(0 , video_length)

        if random_frame not in capture_frame:
            capture_frame[random_frame] = 2

    # 0 hit
    # 1 Prepare Hit
    # 2 Move
    print(f"    HitFrame {len([1 for i in capture_frame.values() if i == 0 ])}")
    print(f"    Prepare Hit {len([1 for i in capture_frame.values() if i == 1 ])}")
    print(f"    MoveFrame {len([1 for i in capture_frame.values() if i == 2 ])}")

    # 參數 - 圖片大小 、 裁切範圍 、 縮放倍率
    img_sizex , img_sizey = config.img_size

    if config.homography != True:
        crop_xt , crop_yt = config.crop_top
        crop_xd , crop_yd = config.crop_bottom
        resize_factor = config.resize_factor
        new_sizex = int(( img_sizex - crop_xt - crop_xd ) / resize_factor )
        new_sizey = int(( img_sizey - crop_yt - crop_yd ) / resize_factor )
    else:
        new_sizex , new_sizey = 330 , 150
        matrix , corner , f = getPerspectiveTransformMatrix(f'{video_folder}/{video_folder.name}.mp4')
        
        if f == -1:
            print("Corner Failed")
            return
        
        corner[0][1] -= 70
        corner[3][1] -= 70
        # print(corner)
        old = np.float32(corner)
        new = np.float32([[0,0], [0,img_sizey-1], [img_sizex-1,img_sizey-1] , [img_sizex-1,0]])
        matrix = cv2.getPerspectiveTransform(old , new)

    
    
    idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx in capture_frame:
            
            if config.homography != True:
                frame = frame[crop_yt : img_sizey-crop_yd , crop_xt : img_sizex - crop_xd]
                frame = cv2.resize(frame, (new_sizex , new_sizey), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(imgs_folder / f"{video_folder.name}_{idx}._{capture_frame[idx]}.jpg"),frame)
            else:
                imgOutput = cv2.warpPerspective(frame, matrix, (img_sizex , img_sizey), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
                imgOutput = cv2.resize(imgOutput, (new_sizex , new_sizey), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(imgs_folder / f"{video_folder.name}_{idx}_{capture_frame[idx]}.jpg"),imgOutput)

        idx += 1
        cv2.waitKey(1)

        ret, frame = cap.read()

    cap.release()

def get_labels_and_frame(video_folder , imgs_folder, labels_folder , config):
    
    check_path = labels_folder / f"{video_folder.name}.csv"

    print(video_folder , str(imgs_folder).split('\\')[1])

    if check_path.exists():
        return

    target_frame_num , capture_frame = hit_label_append(video_folder , labels_folder , config.labels_setting)

    get_frame(video_folder , imgs_folder , capture_frame , target_frame_num , config.imgs_setting)

    
def concat_ball_pos_files(ball_data_folder): 

    all_ball_csv = pd.read_csv("./csv/sample_ball_pos.csv")
    
    for ball_csv in ball_data_folder:
        vid_name = ball_csv.name.split('_')[0]
        
        # print(vid_name)
        try:
            df = pd.read_csv(ball_csv)
            df = df.drop(columns=['Time'])
            df.insert(0 , "VideoName" , [vid_name for i in range(len(df))])

            all_ball_csv = pd.concat([all_ball_csv , df])
        except:
            print(f"{ball_csv.name} Error")
    print(all_ball_csv)

    all_ball_csv.to_csv("./csv/all_ball_pos.csv" , index=False)

def concat_hit_labels_files(vid_folder): 

    all_hit_csv = pd.read_csv("./csv/sample_hit_labels.csv")
    print(all_hit_csv)
    for vid in vid_folder:

        try:
            lables_csv = Path(f"{vid}/{vid.name}_S2.csv")
            df = pd.read_csv(lables_csv)
            
            vid_name = vid.name
            df.insert(0 , "VideoName" , [vid_name for i in range(len(df))])
            all_hit_csv = pd.concat([all_hit_csv , df])
            
            # print(len(df.columns))
            if len(all_hit_csv.columns) == 17:
                print(vid)
                break
        except:
            print(f"{vid} Error")
    
    print(all_hit_csv)
    all_hit_csv.to_csv("all_hit_labels.csv" , index = False)    

if __name__ == '__main__':

    config = Namespace(
        # path setting
        path_setting = Namespace(
            data_path = './data/part1/train/',
            dataset_path = './dataset',
            ball_data_folder = './data/ball_pred',
        ),
        # label setting
        labels_setting = Namespace(
            pick_hit_frame = 1,
            pick_non_hit_frame = [-5,-4,-3 ,-2 ,-1 ,1,2, 3,4,5],
            valid_random_factor = 2,
        ),
        # img setting
        imgs_setting = Namespace(
            homography = True,
            new_image_size = (330,150),
            img_size  = (1280 , 720),
            crop_top = (200 , 100),
            crop_bottom = (200 , 0),
            resize_factor = 4 , 
        )
    )

    path_config = config.path_setting
    data_folder = Path(path_config.data_path).glob('*')
    train_folder = Path(path_config.dataset_path) / 'train'
    valid_folder = Path(path_config.dataset_path) / 'valid'
    ball_data_folder = Path(path_config.ball_data_folder).glob('*.csv')

    (Path(path_config.dataset_path) / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (Path(path_config.dataset_path) / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (Path(path_config.dataset_path) / 'valid' / 'images').mkdir(parents=True, exist_ok=True)
    (Path(path_config.dataset_path) / 'valid' / 'labels').mkdir(parents=True, exist_ok=True)
    


    data_folder_list = [i for i in data_folder]
    ball_data_folder_list = [i for i in ball_data_folder]

    # print(ball_data_folder_list)
    imgs_folder_list = [valid_folder / 'images' if int(i.name.strip('0')) % 10 == 1 else train_folder / 'images' for i in data_folder_list ]
    labels_folder_list = [valid_folder / 'labels' if int(i.name.strip('0')) % 10 == 1 else train_folder / 'labels' for i in data_folder_list ]

    # get_labels_and_frame(data_folder_list[0], imgs_folder_list[0],labels_folder_list[0] , Namespace(labels_setting = config.labels_setting , imgs_setting = config.imgs_setting))
    # get_labels_and_frame(Path("part1/part1/train/00310"), imgs_folder_list[0],labels_folder_list[0] , Namespace(labels_setting = config.labels_setting , imgs_setting = config.imgs_setting))
    # CPU_Core_num = 6
    # pool = Pool(processes = CPU_Core_num)
    # pool.starmap(get_labels_and_frame, zip(
    #         data_folder_list , 
    #         imgs_folder_list , 
    #         labels_folder_list , 
    #         repeat(Namespace(labels_setting = config.labels_setting , imgs_setting = config.imgs_setting))) , 
    #         chunksize = int(len(data_folder_list) / CPU_Core_num))

    concat_ball_pos_files(ball_data_folder_list)
    # concat_hit_labels_files(data_folder_list)




   


