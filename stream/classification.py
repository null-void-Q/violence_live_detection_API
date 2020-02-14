from .transforms import preprocess_input
import numpy as np
import cv2
import datetime
from keras import backend as K
def classify_clip(model,clip,session=None,graph=None):
    #K.set_session(session)
    with graph.as_default():
        with session.as_default():
            processed_clip = []
            for frame in clip:
                processed_frame = preprocess_input(frame)
                processed_clip.append(processed_frame)
            processed_clip = np.expand_dims(processed_clip,axis=0)
            predictions = model.predict(processed_clip, batch_size=len(clip), verbose=0, steps=None)
            predictions = predictions[0]
            return predictions          

def getTopNindecies(array,n):
    sorted_indices = np.argsort(array)[::-1]
    return sorted_indices[:n]
    
def calculate_prediction(predictions, class_map):
    final_prediction= np.zeros((len(class_map)))
    for pred in predictions:
        final_prediction+=pred
    final_prediction/=len(predictions)

    
    top1indices = getTopNindecies(final_prediction,1)
    index = top1indices[0]
    result =  {'label': class_map[index], 'score':round(final_prediction[index]*100,2)} 
    
    return result
        

def write_label(frame, prediction, threshold,alertLabel = 'Violence' , flag = False):
    font= cv2.FONT_HERSHEY_SIMPLEX
    circle_center = (15,15)
    circle_radius = 2
    circle_color = (0,255,0)
    circle_thikcness = 10

    text_location = (20,60)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1

    lineType               = 2
    bordersize             = 10
    borderColor            = (0,0,255)


    label = prediction


    timestamp = datetime.datetime.now()
    cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (32, 64, 255),1)

    if label['label'] == alertLabel:
        
        cv2.putText(frame,label['label']+' Detected', 
            text_location, 
            font, 
            font_scale,
            borderColor,
            lineType,)
        if flag:
            frame = cv2.copyMakeBorder(frame,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[*borderColor])
            
    else:
        cv2.circle(frame, circle_center, circle_radius, circle_color,circle_thikcness, lineType)

    return frame

def anotate_clip(clip,label,threshold):
    out_clip = []
    flag = True
    for i,frame in enumerate(clip):
        out_clip.append(write_label(frame,label,threshold,flag=flag))
        flag = ((i+20) % 20 == 0)
    return  out_clip    
        