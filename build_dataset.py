# from ultralytics import YOLO
from pathlib import Path
import cv2
import pandas as pd
import random
from multiprocessing import Process, Pool
from itertools import repeat


def get_labels_and_frame(video_folder , imgs_folder, labels_folder):
    
    check_path = labels_folder / f"{video_folder.name}.csv"
    print(video_folder)

    if check_path.exists():
        return
    
    ###### lables ######
    labels = pd.read_csv(f"{video_folder}/{video_folder.name}_S2.csv")
    append_label = pd.DataFrame(columns=labels.columns)
    
    
    
    for idx , sublabel in labels[['HitFrame' , 'ShotSeq']].iterrows():
        label = labels.loc[labels['HitFrame'] == sublabel['HitFrame']]
        label = pd.concat([label,label,label]).reset_index(drop=True)
        

        label.at[0, 'HitFrame'] = sublabel['HitFrame'] - 1 
        label.at[0, 'ShotSeq'] = sublabel['ShotSeq'] * 3 - 2
        label.at[1, 'HitFrame'] = sublabel['HitFrame']  
        label.at[1, 'ShotSeq'] = sublabel['ShotSeq'] * 3 - 1
        label.at[2, 'HitFrame'] = sublabel['HitFrame'] + 1 
        label.at[2, 'ShotSeq'] = sublabel['ShotSeq'] * 3 

        append_label = pd.concat([append_label, label])
        

    append_label = append_label.reset_index(drop=True)
    append_label.to_csv(labels_folder / f"{video_folder.name}.csv" , index=False)

    ###### frames ######
    cap = cv2.VideoCapture(f'{video_folder}/{video_folder.name}.mp4')
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    capture_frame = [i for sublist in [[i, i+1 , i-1 , i+10, i-10] for i in labels['HitFrame']] for i in sublist]

    while len(capture_frame) < len(labels['HitFrame']) * 6:
        random_frame = random.randint(0 , video_length)

        if random_frame not in capture_frame:
            capture_frame.append(random_frame)

    capture_frame = sorted(capture_frame)
    
    # 參數
    idx = 0
    img_sizex , img_sizey = 1280 , 720
    crop_xt , crop_yt = 200 , 100
    crop_xd , crop_yd = 200 , 0
    resize_factor = 2
    new_sizex = int(( img_sizex - crop_xt - crop_xd ) / resize_factor )
    new_sizey = int(( img_sizey - crop_yt - crop_yd ) / resize_factor )
    

    # break
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
    train_folder = Path('./part1/part1/train/').glob('*')
    imgs_folder = Path('./dataset/train/images')
    imgs_folder.mkdir(parents=True, exist_ok=True)

    labels_folder = Path('./dataset/train/labels')
    labels_folder.mkdir(parents=True, exist_ok=True)

    train_folder_list = [i for i in train_folder]
    print(len(train_folder_list))

    CPU_Core_num = 6
    pool = Pool(processes = CPU_Core_num)
    pool.starmap(get_labels_and_frame, zip(train_folder_list , repeat(imgs_folder) , repeat(labels_folder)) , chunksize = int(len(train_folder_list) / CPU_Core_num))



