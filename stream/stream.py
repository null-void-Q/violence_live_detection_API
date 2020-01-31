from collections import deque
import cv2
import numpy as np
import threading
import time
from .model import loadModel
from .classification import classify_clip, calculate_prediction, anotate_clip
from .utils import store_clip   
import os

alert_label = 'Violence'


def create_stream(src,labels_path,stream_url,clip_frames = 64 , frame_dims = (224,224,3), prediction_threshold = 50 ,prediction_memory = 4): #stream setup

    labels = [x.strip() for x in open(labels_path)]
    model,session,graph = loadModel(numberOfClasses = len(labels), inputFrames = clip_frames,frameDims= frame_dims,withWeights= 'v_inception_i3d' , withTop= True)
    classified_stream = ClassificationStream(src,labels,model,stream_url,clip_frames=clip_frames,frame_dims = frame_dims
                                ,prediction_memory=prediction_memory,
                                prediction_threshold=prediction_threshold,
                                session=session,
                                graph=graph)

    classified_stream.start_recieving_stream() # return a bool make sure it works
    time.sleep(0.5)
    classified_stream.start_classifying_stream()      # return a bool make sure it works

    return classified_stream

                        
class ClassificationStream:
    def __init__(self,src, label_list, model,stream_url,session= None ,graph=None,clip_frames = 64, frame_dims= (224,224,3),prediction_memory = 4, prediction_threshold = 50):
        self.src = src
        self.label_list = label_list
        self.model= model
        self.session = session
        self.graph = graph
        
        self.stream_url=stream_url

        self.frame_dims = frame_dims 
        self.clip_frames = clip_frames
        self.prediction_memory = prediction_memory
        self.prediction_threshold = prediction_threshold

        self.predictions = deque([])
        self.stream_buffer = deque([])
        self.classified_buffer = deque([])

        self.stop_flag = threading.Event()
        
        self.stream_lock = threading.Lock()
        self.stream_reader = threading.Thread(target=stream_reader, args=(self.src,self.stream_buffer,self.stream_lock,self.stop_flag))
        self.stream_reader.daemon = True

        
        self.classification_lock = threading.Lock()
        self.stream_classifier = threading.Thread(target=classifier, args=(self,))
        self.stream_classifier.daemon = True

    def stream(self,fps = 30): # handel stream prodcasting here
        fail_count = 0
        delay = round(1/fps,2)
        while not self.stop_flag.is_set():

            buffer_length = len(self.classified_buffer)
            
            if buffer_length > 0:
                with self.classification_lock:
                    frame = self.classified_buffer.popleft()
                fail_count = 0
            else:
                if fail_count > 3:
                    ret, buf = cv2.imencode('.jpg', np.zeros((100,100,1)))
                    frame = buf.tobytes()        
                    yield (b'--frame\r\n'
                                 b'Content-Type: image/jpeg\r\n\r\n'+ frame + b'\r\n')
                    self.stop_stream()
                    
                fail_count+=1
                time.sleep(1.0)   # handel buffering
                continue

            time.sleep(delay)      
            ret, buf = cv2.imencode('.jpg', frame)
            frame = buf.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                       
    def stop_stream(self):
    
        self.stop_flag.set()
        self.stream_reader.join(5.0)
        self.stream_classifier.join()
        print('\n')
        print('-'*50)
        print('stream dissconnected: ',self.src)
        exit(1)
    def start_recieving_stream(self):
        self.stream_reader.start()

    def start_classifying_stream(self):
        self.stream_classifier.start()

    def update_predictions(self,prediction):
        self.predictions.append(prediction)
        if len(self.predictions) > self.prediction_memory:
            self.predictions.popleft()


def stream_reader(src,buffer,lock,stop_flag,max_buffer_size = 300): #handel streaming connection and recieving here
    cap = cv2.VideoCapture(src)
    while not stop_flag.is_set():
        if(len(buffer) < max_buffer_size ):
            success, frame = cap.read()
            if success:
                with lock:
                    buffer.append(frame)
            else: # try reconnecting / terminating process here // and in prodcasting
                print('Stream Disconnected: trying to reconnect...')
                time.sleep(1.0)
        else:
            time.sleep(2.0) 
            continue        
        time.sleep(0.01)        

def classifier(stream,max_buffer_size = 100): # handel classification here 
    saveCounter = -1
    maxSaveClipSize = 5
    saveClip = []
    while not stream.stop_flag.is_set():
        if len(stream.classified_buffer) >= max_buffer_size:
            time.sleep(2.0)
            continue

        buffer_length = 0
        clip = []
        buffer_length = len(stream.stream_buffer)
        if buffer_length >= stream.clip_frames:
            with stream.stream_lock:
                for i in range(stream.clip_frames):
                    clip.append(stream.stream_buffer.popleft())
        else:
            time.sleep(0.5) 
            continue

        prediction = classify_clip(stream.model,clip,stream.session,stream.graph)
        stream.update_predictions(prediction)

        label = calculate_prediction(stream.predictions,stream.label_list)
        
        # handel event detected
        ####
        if label['label'] == alert_label: 
            saveCounter = 0
        if saveCounter != -1:
            saveClip.extend(np.copy(clip))
            saveCounter+=1
            if saveCounter > maxSaveClipSize:
                clip_saver = threading.Thread(target=store_clip, args=(saveClip,))
                clip_saver.daemon = True
                clip_saver.start()   
                saveCounter = -1
        #####
        
        clip = anotate_clip(clip,label,stream.prediction_threshold)

        with stream.classification_lock:
            stream.classified_buffer.extend(clip)   