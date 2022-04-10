import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    for filter_name in ['large', 'medium', 'tiny', 'night']:
        flt = np.load(f'{filter_name}.npy')
        plt.imsave(f'{filter_name}.jpg', flt)
