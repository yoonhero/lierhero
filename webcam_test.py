from matplotlib import pyplot as plt
from PIL import Image
import cv2


if __name__ == "__main__":
    WIDTH = 1280
    HEIGHT = 760

    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            cv2.imshow('Result', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("### VIDEO IS ENDED ###")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destoryAllWindows()
