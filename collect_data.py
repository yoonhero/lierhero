import pandas as pd
import time
from utils import LineChart
from webcam import Webcam
# from arduino import ArduinoInput, get_usb_device
from datetime import datetime
import math
import cv2 
# import keyboard
# from pygame.locals import *
# from pynput import keyboard
import os



# OLD
class CollectDataTool():    
    WIDTH, HEIGHT = (680, 400)
    def __init__(self):
        self.dataset = DataSet()
        self.cam = Webcam(0, CollectDataTool.WIDTH, CollectDataTool.HEIGHT)
        # self.sensor = ArduinoInput(get_usb_device()[-1])
        ## TODO: Chart for Data visualizing
        # self.chart = LineChart()

        self.lie = False

        self.rec_condition = False


    def change_lie_condition(self, lie):
        self.lie = lie

    def process(self):
        if not self.rec_condition:
            print("NONO")
            time.sleep(0.1)
            return
        img = self.cam.get_image()

        sensor_values = []
        for _ in range(10):
            v = self.sensor.get_data()
            sensor_values.append(int(v))
            # self.chart.add_value(math.floor(get_timestamp()), v)
            time.sleep(0.1)

        ts = math.floor(get_timestamp())
        
        if not os.path.exists("./dataset"):
            os.makedirs("./dataset")
            os.makedirs("./dataset/lie")
            os.makedirs("./dataset/not_lie")
            

            
        filepath = f"./dataset/lie/{ts}.jpg" if self.lie else f"./dataset/not_lie/{ts}.jpg"

        cv2.imwrite(filepath, img)

        new_row = pd.DataFrame({"img": filepath, "heart_rate": [1, 2, 3, 5, 6, 7, 8,9 , 10, 1], "lie": int(self.lie)})
        self.dataset.add_row(new_row)

        time.sleep(0.1)

    def start(self):
        self.rec_condition = True


        # chart.visualize()
    def end(self, save_file_name):
        self.rec_condition = False
        self.dataset.save(save_file_name)

        self.cam.stop()

        


def on_press(key):
    if key == keyboard.Key.space:
        return True
    return False



    
if __name__ == '__main__':
    collect_data_tool = CollectDataTool()

    collect_data_tool.start()

    time.sleep(2)

    print("Finish Recording")

    collect_data_tool.end("result.csv")
    

    # dataset = DataSet()
    # # dataset.load("dataset.csv")

    # device = 0
    # WIDTH = 640
    # HEIGHT = 400
    # cam = Webcam(device, WIDTH, HEIGHT)

    # sensor = ArduinoInput(get_usb_device()[-1])

    # chart = LineChart()

    # lie = False

    
    # for i in range(3):
    #     lie = False

    #     # with keyboard.Events() as events:
    #     #     for event in events:
    #     #         if event.key == keyboard.Key.space:
    #     #             lie = True
    #     #             print("LIE")
    #     #             break
    #     #         else:
    #     #             break
    #     # key = cv2.waitKey()

    #     # if key == ord('a'):
    #     #     lie = True 


    #     img = cam.get_image()

    #     # if key == keyboard.Key.esc:
    #     #     print("LIE")
    #     #     lie = True

    #     sensor_values = []
    #     for _ in range(10):
    #         v = sensor.get_data()
    #         sensor_values.append(int(v))
    #         chart.add_value(math.floor(get_timestamp()), v)
    #         time.sleep(0.1)

    #     ts = math.floor(get_timestamp())

    #     print("\rts")
        
    #     filepath = f"./dataset/lie/{ts}.jpg" if lie else f"./dataset/not_lie/{ts}.jpg"

    #     cv2.imwrite(filepath, img)

    #     new_row = pd.DataFrame({"img": filepath, "heart_rate": sensor_values, "lie": lie})
    #     dataset.add_row(new_row)

    #     # print(dataset.show_data)

    #     time.sleep(0.1)
    

    # chart.visualize()

    # dataset.save()

    # cam.stop()



