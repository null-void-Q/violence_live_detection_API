from flask import Flask, render_template, Response
import cv2
import time
from stream.stream import create_stream

app = Flask(__name__)

streams = ['./mock_data/1.mp4','./mock_data/2.mp4','https://r2---sn-i5q5g5-1qhl.googlevideo.com/videoplayback?expire=1575932503&ei=933uXfOQD9KKgAftsZiACw&ip=212.106.83.254&id=o-AOXSjvsgwPgJgphZTvjtaakNsW46TPAEiY658T0SxOXV&itag=22&source=youtube&requiressl=yes&mm=31%2C29&mn=sn-i5q5g5-1qhl%2Csn-4g5edne7&ms=au%2Crdu&mv=m&mvi=1&pl=20&initcwndbps=438750&mime=video%2Fmp4&ratebypass=yes&dur=1034.286&lmt=1575810098740145&mt=1575910734&fvip=2&fexp=23842630&c=WEB&txp=5535432&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cmime%2Cratebypass%2Cdur%2Clmt&sig=ALgxI2wwRQIhAK6pvxeXgM_cgtHR3wctuMnXqU8RahRWEQjqndrzTaQCAiBoUOIML13N2I9mboMZoJ9lVLbB0rQodp3NP4Tms-eD9g%3D%3D&lsparams=mm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpl%2Cinitcwndbps&lsig=AHylml4wRQIhAMM2ICuxY_MsrvJ2OXN8yCJTFBwxDtueUS_MQoom8bBtAiAg-OSnWQAMvFhD_FFYVuI0iMQF5-SuIHaO8g0t9x5uQA%3D%3D']
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
