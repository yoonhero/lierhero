import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class LineChart(object):
    def __init__(self, title:str):
        self.x = np.array([], dtype=np.float32)
        self.y = np.array([], dtype=np.float32)

        self.fig = plt.figure(figtitle=title)
        self.ax = self.fig.subplots()

    def add_value(self, x:float, y:float):
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

    def visualize(self, x_label:str, y_label:str):
        # plt.rcParams["font.size"] = 12
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        self.ax.plot(self.x, self.y)
        self.ax.savefig("flot.png", bbox_inches="tight")
