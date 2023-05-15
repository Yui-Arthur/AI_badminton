import pandas as pd
import random
import json


def data_proc(frame_set):
    rel_dic = {}
    for i in frame_set:
        vid , frame , label = i.split('_')
        vid = int(vid)
        frame = int(frame)
        label = int(label)
        f_range = 15
        cond = (all_ball_pos['VideoName'] == vid) & (abs(all_ball_pos['Frame'] - frame) <= f_range)  
        ball_frame = all_ball_pos[cond]

        rel_li = []
        idx = frame - f_range

        for _ , j in ball_frame.iterrows():
            # print(i['Frame'] , idx)
            while idx < j['Frame']:
                rel_li += [0,0,0]
                idx += 1

            if idx == j['Frame']: 
                rel_li += [int(j['Visibility']),int(j['X']),int(j['Y'])]
                idx += 1
        # print("!" , idx , frame + f_range)
        while idx <= frame + f_range:
            rel_li += [0,0,0]
            idx += 1
            # print("!")
        
        rel_dic[i] = rel_li
    return rel_dic

        
all_ball_pos = pd.read_csv("./csv/all_ball_pos_V2_smooth.csv")
all_hit_labels = pd.read_csv("./csv/all_hit_labels.csv")

all_hit_labels = all_hit_labels[['VideoName' , 'HitFrame']]

train_frame_set = set()
valid_frame_set = set()
hit_range = 7


for vid in all_hit_labels['VideoName'].drop_duplicates().values:
    idx = 0
    vid_length = len(all_ball_pos[all_ball_pos['VideoName'] == vid]['Frame'].drop_duplicates())
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

train_data = data_proc(train_frame_set)
valid_data = data_proc(valid_frame_set)

with open('./dataset/train.json' , 'w') as f:
    json.dump(train_data , f)

with open('./dataset/valid.json' , 'w') as f:
    json.dump(valid_data , f)

# with open("train_set.txt" , 'w') as f:
#     for i in train_frame_set:
#         f.write(i+"\n")

# with open("valid_set.txt" , 'w') as f:
#     for i in valid_frame_set:
#         f.write(i+"\n")

# print(len(train_frame_set))
# print(len(valid_frame_set))
# print(vid_hit_dic)
# print(frame_list)