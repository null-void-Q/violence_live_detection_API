from .transforms import preprocess_input
import numpy as np
import cv2
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
        

def write_label(frame, prediction, threshold,alertLabel = 'Violence' ,default_label = {'label':'***', 'score':'***'}):
    font= cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,20)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    bordersize             = 10
    borderColor            = (0,0,255)


    label = prediction

    if label['score'] < threshold:
        label = default_label

    if label['label'] == alertLabel:
       
        frame = cv2.copyMakeBorder(
            frame,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[*borderColor]
        )

    else:
        frame = cv2.putText(frame,label['label'], 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        frame = cv2.putText(
            frame,
            str(label['score'])+'%', 
            (50,50), 
            font, 
            fontScale,
            (0,255,0),
            lineType)   
    return frame
def anotate_clip(clip,label,threshold):
    out_clip = []
    for frame in clip:
        out_clip.append(write_label(frame,label,threshold))
    return  out_clip    