import serial
from serial.tools import list_ports
import time


# ps ax -> for all processes
# fuser /dev/cu.usbmodem141101 -> get process pid
def get_usb_device():
    ports = list(list_ports.comports())

    return [p.device for p in ports]


class ArduinoInput(object):
    def __init__(self, port):
        self.port = port
        self.py_serial = serial.Serial(
            port=self.port,
            baudrate=9600,
        )

    def get_data(self):
        if self.py_serial.readable():
            # 들어온 값이 있으면 값을 한 줄 읽음 (BYTE 단위로 받은 상태)
            # BYTE 단위로 받은 response 모습 : b'\xec\x97\x86\xec\x9d\x8c\r\n'
            response = self.py_serial.readline()

            #  디코딩 후, 출력 (가장 끝의 \n을 없애주기위해 슬라이싱 사용)
            return response[:len(response)-1].decode()


if __name__ == '__main__':
    print(get_usb_device())

    arduino = ArduinoInput(get_usb_device()[-1])

    while True:
        time.sleep(0.1)

        print(arduino.get_data())
