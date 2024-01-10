import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pandas as pd

class StreamingPredictions:
    def __init__(self, master):
        self.master = master
        self.master.title("Data Visualization")
        self.master.configure(bg='#6495ED') 

        self.load_button = tk.Button(self.master, text="Load Database", command=self.load_data, height=2, width=15, background="#4169E1")
        self.load_button.pack(pady=20)

        self.text_label = tk.Label(self.master, text="Streaming Predictions - K-MEANS", font=("Helvetica", 20), background="#6495ED")
        self.text_label.pack(pady=10)

        self.fig = plt.Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        self.musics_columns = None
        self.X = None
        self.y = None

        self.estimators = [
            ("k_means_music_8", KMeans(n_clusters=8, n_init="auto")),
            ("k_means_music_3", KMeans(n_clusters=3, n_init="auto")),
            ("k_means_music_bad_init", KMeans(n_clusters=3, n_init=1, init="random")),
        ]

        self.titles = ["8 clusters", "3 clusters", "3 clusters, bad initialization"]

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            musics_data = pd.read_csv(file_path)
            self.musics_columns = musics_data.columns[1:]
            self.X = musics_data[self.musics_columns].values
            self.y = np.zeros(len(self.X))
            self.plot_clusters()

    def plot_clusters(self):
        self.canvas.figure.clf()
        axes = [self.canvas.figure.add_subplot(2, 2, idx + 1, projection="3d", elev=48, azim=134) for idx in range(3)]

        for (name, est), title, ax in zip(self.estimators, self.titles, axes):
            est.fit(self.X)
            labels = est.labels_

            ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=labels.astype(float), edgecolor="k")

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])
            ax.set_xlabel(self.musics_columns[0])
            ax.set_ylabel(self.musics_columns[1])
            ax.set_zlabel(self.musics_columns[2])
            ax.set_title(title)

        ax_ground_truth = self.canvas.figure.add_subplot(2, 2, 4, projection="3d", elev=48, azim=134)
        for i, genre in enumerate(self.musics_columns):
            if np.any(self.y == i):
                ax_ground_truth.text3D(
                    np.nanmean(self.X[self.y == i, 0]),
                    np.nanmean(self.X[self.y == i, 1]),
                    np.nanmean(self.X[self.y == i, 2]) + 2,
                    genre,
                    horizontalalignment="center",
                    bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
                )

        valid_indices = np.where(~np.isnan(self.X[:, 0]))[0]
        ax_ground_truth.scatter(self.X[valid_indices, 0], self.X[valid_indices, 1], self.X[valid_indices, 2], c=self.y[valid_indices], edgecolor="k")

        ax_ground_truth.xaxis.set_ticklabels([])
        ax_ground_truth.yaxis.set_ticklabels([])
        ax_ground_truth.zaxis.set_ticklabels([])
        ax_ground_truth.set_xlabel(self.musics_columns[0])
        ax_ground_truth.set_ylabel(self.musics_columns[1])
        ax_ground_truth.set_zlabel(self.musics_columns[2])
        ax_ground_truth.set_title("Ground Truth")

        self.canvas.draw()

def main():
    root = tk.Tk()
    app = StreamingPredictions(root)
    root.mainloop()

if __name__ == "__main__":
    main()
