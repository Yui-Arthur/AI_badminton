# from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
import random
from multiprocessing import Process, Pool
from itertools import repeat
import math
from argparse import Namespace


def hit_label_append(video_folder , labels_folder , config):
    ##### hit lables ######
    labels = pd.read_csv(f"{video_folder}/{video_folder.name}_S2.csv")
    append_label = pd.DataFrame(columns=labels.columns)

    # 參數 - 要選幾張連接的打擊圖
    pick_hit_frame = config.pick_hit_frame
    pick_non_hit_frame = config.pick_non_hit_frame
    random_size = len(pick_non_hit_frame) if str(labels_folder).split('\\')[1] == 'train' else len(pick_non_hit_frame) * config.valid_random_factor
    pick_frame = [i - math.floor(pick_hit_frame / 2) for i in range(pick_hit_frame)] + pick_non_hit_frame
    target_frame_num = (pick_hit_frame + len(pick_non_hit_frame) + random_size) *  len(labels['HitFrame'])
    
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

    while len(capture_frame) < target_frame_num:
        random_frame = random.randint(0 , video_length)

        if random_frame not in capture_frame:
            capture_frame[random_frame] = 2

    print(f"    HitFrame {len([1 for i in capture_frame.values() if i == 0 ])}")
    print(f"    NonHitFrame {len([1 for i in capture_frame.values() if i == 1 ])}")
    print(f"    MoveFrame {len([1 for i in capture_frame.values() if i == 2 ])}")

    # 參數 - 圖片大小 、 裁切範圍 、 縮放倍率
    img_sizex , img_sizey = config.img_size
    crop_xt , crop_yt = config.crop_top
    crop_xd , crop_yd = config.crop_bottom
    resize_factor = config.resize_factor
    new_sizex = int(( img_sizex - crop_xt - crop_xd ) / resize_factor )
    new_sizey = int(( img_sizey - crop_yt - crop_yd ) / resize_factor )

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx in capture_frame:
        
            frame = frame[crop_yt : img_sizey-crop_yd , crop_xt : img_sizex - crop_xd]
            frame = cv2.resize(frame, (new_sizex , new_sizey), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(imgs_folder / f"{video_folder.name}_{idx}._{capture_frame[idx]}.jpg"), frame)

        idx += 1
        cv2.waitKey(1)

    cap.release()


def get_labels_and_frame(video_folder , imgs_folder, labels_folder , config):
    
    check_path = labels_folder / f"{video_folder.name}.csv"

    print(video_folder , str(imgs_folder).split('\\')[1])

    if check_path.exists():
        return

    target_frame_num , capture_frame = hit_label_append(video_folder , labels_folder , config.labels_setting)

    get_frame(video_folder , imgs_folder , capture_frame , target_frame_num , config.imgs_setting)

    
def concat_ball_pred_files(ball_data_folder): 

    
    for ball_csv in ball_data_folder:
        df = pd.read_csv(ball_csv)

    pass
if __name__ == '__main__':

    config = Namespace(
        # path setting
        path_setting = Namespace(
            data_path = './part1/part1/train/',
            dataset_path = './dataset',
            ball_data_folder = './ball_pred/ball_pred',
        ),
        # label setting
        labels_setting = Namespace(
            pick_hit_frame = 1,
            pick_non_hit_frame = [-3 ,-2 ,2, 3],
            valid_random_factor = 2,
        ),
        # img setting
        imgs_setting = Namespace(
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

    get_labels_and_frame(data_folder_list[0], imgs_folder_list[0],labels_folder_list[0] , Namespace(labels_setting = config.labels_setting , imgs_setting = config.imgs_setting))
    # CPU_Core_num = 6
    # pool = Pool(processes = CPU_Core_num)
    # pool.starmap(get_labels_and_frame, zip(data_folder_list , imgs_folder_list , labels_folder_list) , chunksize = int(len(data_folder_list) / CPU_Core_num))




   


