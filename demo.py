from __future__ import division, print_function
import numpy as np
from PIL import Image

from pcp import PCP


def bitmap_to_mat(filename):
    #convert img to array
    matrix = []
    shape = None

    img = Image.open(filename).convert("L")
    data = np.asarray(img)
    return data, data.shape


def do_plot(ax, img, shape):
    # plot result
    ax.cla()
    ax.imshow(img.reshape(shape), cmap="gray", interpolation="nearest")
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as pl

    def fit(imgName, dirName, mu=None, lam=None):
        if not os.path.isdir(dirName):
            print(dirName + " is created")
            os.mkdir(dirName)
        M, shape = bitmap_to_mat("data/" + imgName)
        print(imgName + " is fitting. Size = (" + str(shape[0]) + "," +
              str(shape[1]) + ")")

        L, S = PCP(M, mu=mu, lam=lam)  #training

        fig, axes = pl.subplots(1, 3, figsize=(10, 4))
        fig.subplots_adjust(left=0, right=1, hspace=0, wspace=0.01)

        do_plot(axes[0], M, shape)
        axes[0].set_title("raw")
        do_plot(axes[1], M - S, shape)
        axes[1].set_title("low rank")
        do_plot(axes[2], S, shape)
        axes[2].set_title("sparse")
        fig.savefig(dirName + "/{0}".format(imgName))
        print(imgName + " is finished.")

    muList = [1, 10, 100, 1000]

    for m in muList:
        for img in os.listdir("data"):
            fit(img, str(m) + "reults", m)
            #fit(img, "test", m)