from flask import (
    Flask, render_template,jsonify, Response
)
import time
import math
import cv2
import pandas as pd
import numpy as np

from .dataset import CustomDataFrame
from .charts import LineChart
from .utils import get_timestamp, create_directory
from .arduino import ArduinoInput, get_usb_device
from .webcam import Webcam


app = Flask(__name__)

dataset = CustomDataFrame()

WIDTH, HEIGHT = (1080, 760)
cam = Webcam(0, WIDTH, HEIGHT)

rec_condition = False
lie = False

sensor = ArduinoInput(get_usb_device()[-1])

visualize_chart = LineChart("Heart Rate Sensor Tracking")

directories = ["./dataset", "./dataset/lie", "./dataset/not_lie"]


## Generate Frame And Processing Frame when Webcam source IN.
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

            # if directory is empty and create it.
            create_directory(directories)
            
            filepath = f"./dataset/lie/{ts}.jpg" if lie else f"./dataset/not_lie/{ts}.jpg"

            cv2.imwrite(filepath, frame)

            # if lie:
            #     sensor_values = [60, 70, 80, 100, 90, 100, 80, 78, 75, 74]
            # else:
            #     sensor_values = [5, 6, 1, 2, 4, 6, 7, 9, 1, 10]
            sensor_values = np.array([], dtype=np.float32)
            for _ in range(10):
                v = sensor.get_data()
                sensor_values.append(int(v))
                time.sleep(0.1)

            # Append Mean of the 10 sensor values
            visualize_chart.add_value(sensor_values.mean())

            heart_rate = "|".join([str(v) for v in sensor_values])

            # filepath: STR | Heart Rate: np.array (1, 10) | Lie : 1 or 0 (Bool as INT)
            new_row = pd.DataFrame({"image": filepath, "heart_rate":heart_rate, "lie": int(lie)}, index=[len(dataset)])
            dataset.add_row(new_row)

            time.sleep(0.1)

        ret,buffer=cv2.imencode('.jpg',frame)

        assert ret, "Encoding A Frame Fail" 

        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Main Page
@app.route('/', methods=["GET"])
def index():
    return render_template('base.html')

# Video Wrapper API
@app.route("/video")
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


# Change Lie Condition API
@app.route("/lie/<condition>", methods=["GET"])
def change_lie_condition(condition):
    try:
        global lie
        lie = bool(int(condition))

        return jsonify({"message": "Success to change lie status!!"})
    except:
        return jsonify({"message": "Fail to change lie status.."})

# Change Rec Condition API
@app.route("/rec/<condition>", methods=["GET"])
def change_rec_condition(condition):
    try: 
        global rec_condition
        rec_condition = bool(int(condition))

        if not rec_condition:
            dataset.save("./data.csv")

        return jsonify({"message":"Success to change recording status."})
    except:
        return jsonify({"message":"Fail to change recording status.."})


@app.route("/visualize", methods=["GET"])
def visualize():
    visualize_chart.visualize("Time", "Heart Rate")



if __name__ == "__main__":
    app.run(host = '127.0.0.1', port=8080)