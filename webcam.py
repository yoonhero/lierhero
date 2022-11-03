import cv2
import time

class Webcam():
    def __init__(self, device:int, width:int, height:int):
        self.cap = cv2.VideoCapture(device)
        self.WIDTH = width
        self.HEIGHT = height

        self.init_camera()

    def init_camera(self):
        ramp_frames = 30

        for _ in range(ramp_frames):
            _ = self.cap.read()

    def get_image(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()

            assert ret, "Problem when reading WebCam"
            
            return frame

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    WIDTH = 1280
    HEIGHT = 760

    cam = Webcam(0, WIDTH, HEIGHT)

    img = cam.get_image()

    cv2.imwrite('testimg.jpg', img)

    cv2.imshow("Result", img)

    time.sleep(1)

    cam.stop()

    # cap = cv2.VideoCapture(1)

    # while cap.isOpened():
    #     ret, frame = cap.read()

    #     if ret:
    #         cv2.imshow('Result', frame)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         print("### VIDEO IS ENDED ###")

    #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # cap.release()
    # cv2.destoryAllWindows()
