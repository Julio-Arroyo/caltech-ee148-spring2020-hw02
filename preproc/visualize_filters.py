import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    for i in range(4):
        flt = np.load(f'preproc/filter{i}.npy')
        plt.imsave(f'preproc/filter{i}.jpg', flt)
