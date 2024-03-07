from turtle import window_width
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

'''
Utils for visualization of loss curve, image slice...

1. pytorch_lightning.Loggers.csv_logger  metrics helpers.
'''
def inspect_metrics(path):
    df = pd.read_csv(path)
    print("keys in ", path, ":")
    for col in df.columns:
        print(col)
    print('')

def load_step_curve(path, key):
    df = pd.read_csv(path)
    if 'step' in df.columns:
        _data = df.loc[:, ['step', key]]
        _data = _data.dropna()
        _data = _data.set_index(['step'])
    else:
        _data = df.loc[:, key]
        _data = _data.dropna()
    # print(_data.head(60))
    return _data

def load_epoch_curve(path, key):
    df = pd.read_csv(path)
    if 'epoch' in df.columns:
        _data = df.loc[:, ['epoch', key]]
        _data = _data.dropna()
        _data = _data.set_index(['epoch'])
    else:
        _data = df.loc[:, key]
        _data = _data.dropna()
        # _data = _data.set_index(['epoch'])
    # print(_data.head(60))
    return _data

def smooth_curve(data: np.ndarray, ratio=0.5, quiet=True):
    # smooth data using moving average, similar to smoothing factor in tensorboard
    window_width = int(data.shape[0] * ratio)
    half_window_l = window_width // 2
    half_window_r = window_width - half_window_l
    if not quiet: print('window width: ', window_width)
    cumsum = np.cumsum(data)

    # print(data[:10])
    # print(cumsum[:10])

    cumsum_a = np.concatenate([np.zeros(window_width), cumsum])
    cumsum_b = np.concatenate([cumsum, np.ones(window_width) * cumsum[-1]])
    cumsum_complete = (cumsum_b - cumsum_a)[half_window_l: -half_window_r]

    window_width = np.concatenate([np.arange(window_width)+1, np.ones(data.shape[0] - window_width) * window_width, window_width -1 - np.arange(window_width)])[half_window_l: -half_window_r]

    # print(cumsum_complete[:10])
    # print(window_width[:10])

    # print(cumsum_complete.shape, window_width.shape)
    smooth = cumsum_complete / window_width
    # if not quiet: print('data length: {}, smooth length: {}'.format(data.shape[0], smooth.shape[0]))
    # return smooth
    return smooth


'''3D visualization helper'''
class slice_viewer_X:
    def __init__(self, ax, X, step=3):
        self.ax = ax
        # ax.set_title('slice viewer')
        self.step = step

        self.X = X
        # rows, cols, self.slices = X.shape
        self.slices, rows, cols = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap=plt.get_cmap('gray'))
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + self.step) % self.slices
        else:
            self.ind = (self.ind - self.step) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def slice_visualize_X(X, step=3):
    fig, ax = plt.subplots(1, 1)
    tracker = slice_viewer_X(ax, X, step=step)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

class slice_viewer_XY:
    def __init__(self, ax, X, Y):
        self.ax = ax
        # ax.set_title('slice viewer')

        self.X = X
        self.Y = Y
        # rows, cols, self.slices = X.shape
        self.slices, rows, cols = X.shape
        self.ind = None
        # for i in reversed(range(self.slices)):
        #     if np.sum(self.Y[:, :, i]) > 0:
        #         self.ind = i
        #         break
        # assert self.ind is not None, "viewer receives null volume"
        self.ind = self.slices//2
        # self.ind = 487

        self.im = ax[0].imshow(self.X[self.ind, :, :])
        self.label = ax[1].imshow(self.Y[self.ind, :, :])
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 3) % self.slices
        else:
            self.ind = (self.ind - 3) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.label.set_data(self.Y[self.ind, :, :])
        self.ax[0].set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        self.label.axes.figure.canvas.draw()

def slice_visualize_XY(X, Y):
    fig, ax = plt.subplots(1, 2)
    tracker = slice_viewer_XY(ax, X, Y)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

class slice_viewer_X2Y2:
    def __init__(self, ax, X, Y, X2, Y2):
        self.ax = ax
        # ax.set_title('slice viewer')

        self.X = X
        self.X2 = X2
        self.Y = Y
        self.Y2 = Y2
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax[0, 0].imshow(self.X[self.ind,:, :], cmap='gray')
        self.im2 = ax[1, 0].imshow(self.X2[self.ind,:, :], cmap='gray')
        self.label = ax[0, 1].imshow(self.Y[self.ind,:, :], cmap='gray')
        self.label2 = ax[1, 1].imshow(self.Y2[self.ind,:, :], cmap='gray')
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 3) % self.slices
        else:
            self.ind = (self.ind - 3) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind,:, :])
        self.im2.set_data(self.X2[self.ind,:, :])
        self.label.set_data(self.Y[self.ind,:, :])
        self.label2.set_data(self.Y2[self.ind,:, :])
        self.ax[0,0].set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()
        self.label.axes.figure.canvas.draw()
        self.label2.axes.figure.canvas.draw()

def slice_visualize_X2Y2(X, Y, X2, Y2):
    fig, ax = plt.subplots(2, 2)
    tracker = slice_viewer_X2Y2(ax, X, Y, X2, Y2)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

def visualize_X2Y2(X, Y, X2, Y2):
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(X)
    ax[1, 0].imshow(X2)
    ax[0, 1].imshow(Y)
    ax[1, 1].imshow(Y2)
    plt.show()

def visualize_XY(X, Y, axis='off'):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(X, cmap=plt.get_cmap('gray'))
    ax[1].imshow(Y, cmap=plt.get_cmap('gray'))
    # ax[0].axis(axis)
    # ax[1].axis(axis)
    plt.show()

def visualize_XYZ(X, Y, Z):
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(X)
    ax[1].imshow(Y)
    ax[2].imshow(Z)
    plt.show()

if __name__ == '__main__':
    ckpt_path = r'D:\Chang\topogram_recon\lightning_logs\x2ct_pl\version_413543'
    metrics_log = Path(ckpt_path) / 'metrics.csv'
    seg_loss = load_step_curve(str(metrics_log), 'seg_loss')
    data = seg_loss['seg_loss'].to_numpy()
    # data = np.ones(1000)
    s = smooth_curve(data, ratio=0.5)
    plt.plot(s)
    plt.show()
    
