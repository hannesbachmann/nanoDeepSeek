import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time


class AnimatedPlot:
    def __init__(self, root):
        self.root = root
        self.root.title("Animated Plot")

        self.fig, self.ax = plt.subplots()
        self.values = []
        self.cycles = []
        self.line, = self.ax.plot([], [], label='function')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_plot(self, values):
        cycle = len(self.values)
        self.values = values
        self.cycles.append(cycle)

        self.line.set_data(self.cycles, self.values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


class AnimatedHeatmap:
    def __init__(self, root):
        self.root = root
        self.root.title("Animated Plot")

        self.fig, self.ax = plt.subplots()
        self.values = np.zeros((16, 384))
        self.map = self.ax.imshow(self.values, cmap='viridis')
        self.cbar = self.ax.figure.colorbar(self.map, ax=self.ax)
        self.ax.set_aspect('auto')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def set_labels(self, x_ticks, y_ticks):
        self.ax.set_xticks(range(len(x_ticks)), labels=x_ticks,
                           rotation=45, ha="right", rotation_mode="anchor")
        self.ax.set_yticks(range(len(y_ticks)), labels=y_ticks)

    def update_plot(self, values):
        self.values = values

        self.map.set_array(self.values)
        self.map.set_clim(vmin=np.min(self.values), vmax=np.max(self.values))
        self.canvas.draw_idle()

# === Example usage ===

import numpy as np

# Create the GUI


def example(gui):
    # Simulate a training loop that runs separately
    losses = []
    for i in range(1000):
        losses.append(np.exp(-i / 20) + np.random.normal(scale=0.02))  # fake loss value
        gui.update_plot(losses)
        time.sleep(0.1)  # simulate time-consuming training step


# if __name__ == '__main__':
#     root = tk.Tk()
#     gui = AnimatedPlot(root)
#
#     threading.Thread(target=example, args=(gui,), daemon=True).start()
#
#     # Start the GUI loop in the main thread
#     root.mainloop()
