from collections import deque
import cv2
import numpy as np
import threading
import time
from .model import loadModel
import tensorflow as tf
import keras
from .classification import classify_clip, calculate_prediction, anotate_clip
   

def create_stream(src,labels_path,clip_frames = 16 , frame_dims = (224,224,3), prediction_threshold =30 ,prediction_memory = 5): #stream setup

    labels = [x.strip() for x in open(labels_path)]

    model = loadModel(numberOfClasses = len(labels), inputFrames = clip_frames, frameDims = frame_dims,withWeights= True , withTop= True) 
    
    classified_stream = ClassificationStream(src,model,labels,clip_frames=clip_frames
                                ,prediction_memory=prediction_memory,
                                prediction_threshold=prediction_threshold)

    classified_stream.start_recieving_stream() # return a bool make sure it works
    classified_stream.start_classifying_stream()      # return a bool make sure it works

    return classified_stream


def stream_reader(src,buffer,lock): #handel streaming connection and recieving here
    cap = cv2.VideoCapture(src)
    while True:
        success, frame = cap.read()
        if success:
            success,frame=cap.read()
            with lock:
                buffer.append(frame)
        else: # try reconnecting / terminating process here // and in prodcasting
            print('Stream Disconnected: trying to reconnect...')
            time.sleep(0.5)
        time.sleep(0.005)        

def classifier(stream): # handel classification here

    while True:
        buffer_length = 0
        clip = []
        with stream.stream_lock:
            buffer_length = len(stream.stream_buffer)
            if buffer_length >= stream.clip_frames:
                for i in range(stream.clip_frames):
                    clip.append(stream.stream_buffer.popleft())
            else: continue

        prediction = classify_clip(stream.model,clip)
        stream.update_predictions(prediction)

        label = calculate_prediction(stream.predictions,stream.label_list)
        anotate_clip(clip,label,stream.prediction_threshold)

        with stream.classification_lock:
            stream.classified_buffer.extend(clip)   
                        

class ClassificationStream:
    def __init__(self,src, model, label_list, clip_frames = 16, prediction_memory = 5, prediction_threshold = 30):
        self.src = src
        self.model = model
        self.label_list = label_list

        self.clip_frames = clip_frames
        self.prediction_memory = prediction_memory
        self.prediction_threshold = prediction_threshold

        self.predictions = deque([])
        self.stream_buffer = deque([])
        self.classified_buffer = deque([])

        self.stream_lock = threading.Lock()
        self.stream_reader = threading.Thread(target=stream_reader, args=(self.src,self.stream_buffer,self.stream_lock))
        self.stream_reader.daemon = True

        
        self.classification_lock = threading.Lock()
        self.stream_classifier = threading.Thread(target=classifier, args=(self,))
        self.stream_classifier.daemon = True

    def stream(self,delay): # handel stream prodcasting here
        while True:
            buffer_length = 0
            with self.classification_lock:
                buffer_length = len(self.classified_buffer)
                if buffer_length > 0:
                    frame = self.classified_buffer.popleft()
                else: continue
            time.sleep(delay)      
            ret, buf = cv2.imencode('.jpg', frame)
            frame = buf.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')        

    def start_recieving_stream(self):
        self.stream_reader.start()

    def start_classifying_stream(self):
        self.stream_classifier.start()

    def update_predictions(self,prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > self.prediction_memory:
            self.predictions.popleft()
