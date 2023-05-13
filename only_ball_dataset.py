import pandas as pd
import random

all_ball_pos = pd.read_csv("./csv/all_ball_pos_V2.csv")
all_hit_labels = pd.read_csv("./csv/all_hit_labels.csv")

all_hit_labels = all_hit_labels[['VideoName' , 'HitFrame']]

train_frame_set = set()
valid_frame_set = set()
hit_range = 5


for vid in all_hit_labels['VideoName'].drop_duplicates().values:
    idx = 0
    vid_length = len(all_ball_pos[all_ball_pos['VideoName'] == vid])
    vid_hit_frame = all_hit_labels[all_hit_labels['VideoName'] == vid]['HitFrame'].values
    pick_num = len(vid_hit_frame) * (hit_range * 2 + 1)
    # print(len(vid_hit_frame))

    for frame in vid_hit_frame:
        for r in range(-hit_range , hit_range+1):
            # print(r)
            if vid % 10 == 1:
                valid_frame_set.add(f"{vid}_{frame + r}_0")
            else:
                train_frame_set.add(f"{vid}_{frame + r}_0")

    while idx < pick_num and idx + pick_num < vid_length:
        pick_frame = random.randint(1, vid_length + 1)
        if pick_frame not in vid_hit_frame:
            if vid % 10 == 1:
                valid_frame_set.add(f"{vid}_{pick_frame}_1")
            else:
                train_frame_set.add(f"{vid}_{pick_frame}_1")
            idx+=1



print(len(train_frame_set))
print(len(valid_frame_set))
# print(vid_hit_dic)
# print(frame_list)