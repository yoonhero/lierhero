import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm

class LineChart(object):
    def __init__(self):
        self.x = np.array([], dtype=np.int64)
        self.y = np.array([], dtype=np.int64)

    def add_value(self, x, y):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

    def visualize(self):
        plt.rc('font', family='NanumMyeongjo')
        plt.rcParams["font.size"] = 12

        plt.ylabel('Heart Rate')
        plt.xlabel('Timestamp')

        plt.plot(self.x, self.y)
        plt.show()

