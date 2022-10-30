from flask import (
    Flask, redirect, render_template, request, session,jsonify, url_for, Response
)
import time
from collect_data import DataSet, get_timestamp
from utils import LineChart, create_directory
from arduino import ArduinoInput, get_usb_device
from webcam import Webcam
import math
import cv2
import pandas as pd
import torch
import random

app = Flask(__name__)

dataset = DataSet()

WIDTH, HEIGHT = (680, 400)
cam = Webcam(0, WIDTH, HEIGHT)



rec_condition = False
lie = False

sensor = ArduinoInput(get_usb_device()[-1])


def generate_frames():
    while True:
        global rec_condition
        global lie
        global dataset
        global sensor
            
        ## read the camera frame
        frame = cam.get_image()
        frame = cv2.flip(frame, 1)
        if rec_condition:
            ts = math.floor(get_timestamp())

            create_directory()
            
            filepath = f"./dataset/lie/{ts}.jpg" if lie else f"./dataset/not_lie/{ts}.jpg"

            cv2.imwrite(filepath, frame)

            if lie:
                sensor_values = [60, 70, 80, 100, 90, 100, 80, 78, 75, 74]
            else:
                sensor_values = [5, 6, 1, 2, 4, 6, 7, 9, 1, 10]
            #for _ in range(10):
            #    v = sensor.get_data()
            #    sensor_values.append(int(v))
            #    ## TODO: Chart
            #    time.sleep(0.1)

            heart_rate = "|".join([str(v) for v in sensor_values])

            new_row = pd.DataFrame({"image": filepath, "heart_rate":heart_rate, "lie": int(lie)}, index=[len(dataset)])
            dataset.add_row(new_row)

            time.sleep(0.1)

        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()


        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=["GET"])
def index():

    return render_template('base.html')


@app.route("/video")
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/lie/<condition>", methods=["GET"])
def change_lie_condition(condition):
    try:
        global lie
        lie = bool(int(condition))

        return jsonify({"message": "Success!!"})
    except:
        return jsonify({"message": "Fail..."})

@app.route("/rec/<condition>", methods=["GET"])
def change_rec_condition(condition):
    try: 
        global rec_condition
        rec_condition = bool(int(condition))

        if not rec_condition:
            dataset.save("./data.csv")

        return jsonify({"message":"Success!!"})
    except:
        return jsonify({"message":"Fail..."})



if __name__ == "__main__":
    app.run(host = '127.0.0.1', port=8080)