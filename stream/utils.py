import cv2
import os
import numpy as np
from datetime import datetime

def store_clip(clip, storeDirectory = './recorded_clips/'):

    if not os.path.exists(storeDirectory):
        os.makedirs(storeDirectory)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    clip_name = 'PV_'+dt_string+'.mp4'
    out = cv2.VideoWriter(storeDirectory+'/'+clip_name,cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,480))# 30.0 fps
    for frame in clip:
        frame = cv2.resize(frame,(640,480))
        out.write(frame) 
    out.release()