
def guess_hitter(ball_labels , hit_labels):
    

    ball_labels = ball_labels.drop(ball_labels[ball_labels['Visibility'] == 0].index)


    odd_labels = hit_labels[(hit_labels['ShotSeq'] % 2 == 1)]['HitFrame']
    even_labels = hit_labels[(hit_labels['ShotSeq'] % 2 == 0)]['HitFrame']
    

    frame_range = 3
    odd_hit_range = [i+j for i in odd_labels.values for j in range(-frame_range,frame_range+1)]
    even_hit_range = [i+j for i in even_labels.values for j in range(-frame_range,frame_range+1)]
    
    odd_ball_labels = ball_labels[(ball_labels['Frame'].isin(odd_hit_range))]['Y'].values
    even_ball_labels = ball_labels[(ball_labels['Frame'].isin(even_hit_range))]['Y'].values
                                #    


    y_range = 0
    if((odd_ball_labels.mean() - y_range  > even_ball_labels.mean())):
        return "B"
    elif ((odd_ball_labels.mean() - y_range  < even_ball_labels.mean())):
        return "A"
    else:
        return -1
        