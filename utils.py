import os
from datetime import datetime

def get_timestamp():
    dt = datetime.now()

    ts = datetime.timestamp(dt)

    return ts


def create_directory(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

            