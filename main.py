from flask import Flask, render_template, Response, request, make_response
import cv2
import time
from stream.stream import create_stream
import json

app = Flask('VD')

streams = {}

@app.route('/start_stream/', methods=['POST', 'GET'])    
def start_stream():
    global streams
    if request.method == 'POST':
        streamId = request.form["stream_id"]
        streamSource = request.form["stream_source"]
        try: # TODO handel stream source error
            cs = create_stream(streamSource,'./txt/violence_labels.txt')
            streams[str(streamId)] = cs
            resp = make_response({'message':'stream started.', 'stream_url':'/video_feed/'+str(streamId)}, 200)
        except Exception as e:
            print(e)
            resp = make_response({'error':'Starting Stream Failed!'}, 400)

        resp.headers["Access-Control-Allow-Origin"] = "*"        
        return resp
        

@app.route('/video_feed/<id>/')
def video_feed(id):
    global streams

    if not id in streams:
        resp = make_response({'error':'Stream Not Found.'}, 404)
        resp.headers["Access-Control-Allow-Origin"] = "*"   
        return resp

    cs = streams[id]
    return Response(cs.stream(fps=30),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="7000")
