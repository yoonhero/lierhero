import torch 
import cv2
import mediapipe as mp
from arduino import ArduinoInput, get_usb_device
import time


# TODO: Implementation Prediction

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh

sensor = ArduinoInput(get_usb_device()[-1])

model = torch.load("./weights/model_v1.pt")


def prediction(model, x1, x2):
    pred = model(x1, x2)
    return (pred >= torch.FloatTensor([0.5]))[0]


with mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
    
        lie = False

        if not success:
            print("Ignoring empty camera frame")

            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        temp_landmark = face_mesh.process(image)
        try:
            tensor_landmark = torch.tensor([[[landmark.x, landmark.y, landmark.z] for landmark in list(temp_landmark.multi_face_landmarks[0].landmark)]], dtype=torch.float32)
        except: 
            tensor_landmark = torch.randn((1, 478, 3))
        # sensor_values = []
        # for _ in range(10):
        #     v = sensor.get_data()
        #     sensor_values.append(int(v))
        #     time.sleep(0.1)

        
        x1 = tensor_landmark
       # x2 = torch.tensor([sensor_values], dtype=torch.float32)
        x2 = torch.randn((1, 10), dtype=torch.float32)

        pred = prediction(model, x1, x2)

        if pred:
            lie = True

        if lie:
            text = "You're lying!!"
        else: 
            text = "..."

        org=(50,100)
        font=cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame,text,org,font,1,(255,0,0),2)

        cv2.imshow("LierHero", frame)


cv2.waitKey()
cv2.destroyAllWindows()