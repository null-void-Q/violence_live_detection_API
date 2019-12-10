from flask import Flask, render_template, Response
import cv2
import time
from stream.stream import create_stream
app = Flask(__name__)

streams = ['./mock_data/1.mp4','./mock_data/2.mp4','./mock_data/3.mp4']
def get_stream(id):
    try:
    	return streams[int(id)]
    except:
    	print('*** Error getting stream: No stream for provided id')
def add_stream(name):
    streams.append(name) # needs validation


@app.route('/video_feed/<id>/', methods=["GET"])
def video_feed(id):

    cs = create_stream(get_stream(id),'./txt/label_map.txt')
    time.sleep(2.0) # give stream a head start

    return Response(cs.stream(delay=0.05),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="7000")
