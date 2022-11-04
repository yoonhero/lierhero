from flask import (
    Flask, render_template, Response
)
import torch 
import cv2
import mediapipe as mp

from arduino import ArduinoInput, get_usb_device
from webcam import Webcam


app = Flask(__name__)

WIDTH, HEIGHT = (1080, 760)
cam = Webcam(0, WIDTH, HEIGHT)

sensor = ArduinoInput(get_usb_device()[-1])


mp_face_mesh = mp.solutions.face_mesh
face_detector = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4)


model = torch.load("./weights/model_v1-2.pt")

def prediction(x1, x2):
    pred = model(x1, x2)
    return (pred >= torch.FloatTensor([0.5]))[0]

## Generate Frame And Processing Frame when Webcam source IN.
def generate_frames():
    while True:
        ## read the camera frame
        frame = cam.get_image()
        frame = cv2.flip(frame, 1)

        lie = False

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        temp_landmark = face_detector.process(image)
        try:
            tensor_landmark = torch.tensor([[landmark.x, landmark.y, landmark.z] for landmark in list(temp_landmark.multi_face_landmarks[0].landmark)], dtype=torch.float32)
        except: 
            tensor_landmark = torch.randn((478, 3))
        
        x1 = tensor_landmark.unsqueeze(0)

        # TODO: Sensor Value Expectation
        # sensor_values = []
        # for _ in range(10):
        #     v = sensor.get_data()
        #     sensor_values.append(int(v))
        #     time.sleep(0.001)
       # x2 = torch.tensor([sensor_values], dtype=torch.float32)
        x2 = torch.randn((1, 10), dtype=torch.float32)

        pred = prediction(x1, x2)

        # print(f"LIE {pred}")

        if pred:
            lie = True

        if lie:
            text = "You're lying!!"
        else: 
            text = "..."

        org=(50,100)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,text,org,font,4,(255,0,0),8)
    

        ret,buffer=cv2.imencode('.jpg',frame)

        assert ret, "Encoding A Frame Fail" 

        frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')

# Video Wrapper API
@app.route("/video")
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host = '127.0.0.1', port=8080)