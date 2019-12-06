from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import cv2
import numpy as np
import sys
import time
from i3d_inception import Inception_Inflated3d
from argparse import ArgumentParser
from transforms import preprocess_input
from collections import deque 
from model import loadModel



outputFrame = None
lock = threading.Lock()
clipDuration = 16
memory =  5 
threshold = 30
 
# initialize a flask object
app = Flask("STREAM-TEST")
 
cap = cv2.VideoCapture(0)
time.sleep(2.0)




@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")






def classify_live(classes_list,model):
	# grab global references to the video stream, output frame, and
	# lock variables
	global cap, outputFrame, lock


	total = 0    

	preds = deque([])
	clip = []

	defaultPred = {'label':'----', 'score':'----'}
	prediction = {'label':'----', 'score':0.0}

	preds_count = 0
	i = 0
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			labeled = np.copy(frame)
			write_label(labeled,prediction,threshold,defaultPred,classes_list)
			frame = preprocess_input(frame)
			clip.append(frame)  
			i+=1
			if(i == clipDuration):
				preds.append(classify_clip(clip,model))
				prediction = calculate_prediction(preds,classes_list)
				preds_count += 1
				clip=[]
				i=0
			if preds_count == memory:
				preds.popleft()
				preds_count = memory-1 
		# acquire the lock, set the output frame, and release the
		# lock
			with lock:
				outputFrame = labeled   	  
		else:
				cap.set(cv2.CAP_PROP_POS_FRAMES,0)
				
		key = cv2.waitKey(10)
		if key == ord('q'):
			break


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
    print(result)
    return result

def classify_clip(clip, model):

    clip = np.expand_dims(clip,axis=0)
    out_logits = model.predict(clip, batch_size=len(clip), verbose=0, steps=None)
    predictions = out_logits
    predictions = predictions[0]
    return predictions                  

def write_label(frame, prediction, threshold, defaultPred, classesSubset):
    font= cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,20)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    label = prediction

    #if label['score'] < threshold or not label['label'] in classesSubset :
        #label = defaultPred

    cv2.putText(frame,label['label'], 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.putText(
        frame,
        str(label['score'])+'%', 
        (50,50), 
        font, 
        fontScale,
        (0,255,0),
        lineType)   


def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
 
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
 
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpeg", outputFrame)
 
			# ensure the frame was successfully encoded
			if not flag:
				continue
 
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

	
# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
 


	labels = [x.strip() for x in open('label_map.txt')]
	model = Inception_Inflated3d(include_top=True,
                                        weights='rgb_inception_i3d',
                                        input_shape=(clipDuration,224,224,3),
                                        classes=400,
                                        endpoint_logit=False)
	model._make_predict_function()
	# start a thread that will perform motion detection
	t = threading.Thread(target=classify_live, args=(labels,model,))
	t.daemon = True
	t.start()
 
	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
 
# release the video stream pointer
cap.release()