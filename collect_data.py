import pandas as pd
import time
from utils import LineChart
from webcam import Webcam
from arduino import ArduinoInput, get_usb_device
from datetime import datetime
import math
import cv2
import keyboard
from pygame.locals import *
from pynput import keyboard



def get_timestamp():
    dt = datetime.now()

    ts = datetime.timestamp(dt)

    return ts



class DataSet(object):
    def __init__(self):
        self.dataframe = pd.DataFrame(columns = ['img', 'heart_rate', 'lie'])

    def load(self, filepath):
        self.dataframe = pd.read_csv(filepath)

    def add_row(self, row:dict):
        self.dataframe = self.dataframe.append(row, ignore_index=True)

    def show_data(self):
        return self.dataframe

    def save(self):
        self.dataframe.to_csv("dataset.csv", index=False)

def on_press(key):
    if key == keyboard.Key.space:
        return True
    return False
    
if __name__ == '__main__':
    

    dataset = DataSet()
    # dataset.load("dataset.csv")

    device = 0
    WIDTH = 640
    HEIGHT = 400
    cam = Webcam(device, WIDTH, HEIGHT)

    sensor = ArduinoInput(get_usb_device()[-1])

    chart = LineChart()

    lie = False

    
    for i in range(3):
        lie = False

        with keyboard.Events() as events:
            for event in events:
                if event.key == keyboard.Key.space:
                    lie = True
                    print("LIE")
                    break
                else:
                    break


        img = cam.get_image()

        # if key == keyboard.Key.esc:
        #     print("LIE")
        #     lie = True

        sensor_values = []
        for _ in range(10):
            v = sensor.get_data()
            sensor_values.append(int(v))
            chart.add_value(math.floor(get_timestamp()), v)
            time.sleep(0.1)

        ts = math.floor(get_timestamp())
        
        filepath = f"./dataset/lie/{ts}.jpg" if lie else f"./dataset/not_lie/{ts}.jpg"

        cv2.imwrite(filepath, img)

        new_row = pd.DataFrame({"img": filepath, "heart_rate": sensor_values, "lie": lie})
        dataset.add_row(new_row)

        # print(dataset.show_data)

        time.sleep(0.1)
    

    chart.visualize()

    dataset.save()

    cam.stop()

