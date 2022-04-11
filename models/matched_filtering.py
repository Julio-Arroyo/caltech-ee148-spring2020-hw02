from models.layers import compute_convolution
import matplotlib.pyplot as plt
import numpy as np


# IDEA: find lowest row coordinate of red light in training set, and only search above
# hyperparameter to only search given top fraction of image,
# since most traffic lights will be higher, and lower you will mostly
# find tail lights from cars
TOP_FRAC = 0.8


def predict_boxes(heatmap, filter_dims, conf_thr=0.9):
    '''
    This function takes heatmap, and finds regions with confidence scores
    higher than conf_thr, returning a list of bounding boxes.

    filter_dims - tuple (n_rows, n_cols) of filter used to create heatmap
    '''
    (f_rows, f_cols) = filter_dims

    bboxes = []  # format: TLrow, TLcol, BRrow, BRcol, conf score

    i = 0
    while i < heatmap.shape[0]:
        j = 0
        big_skip = False
        while j < heatmap.shape[1]:
            if heatmap[i][j] > conf_thr:
                bboxes.append([i, j, i + f_rows, j + f_cols, heatmap[i][j]])
                big_skip = True
                j += 9
            j += 1
        if big_skip:
            i += 10
        else:
            i += 1

    return bboxes


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    # IDEA: use scanning to search for red, then used matched filtering at those locations

    # You may use multiple stages and combine the results
    T = np.load('preproc/night.npy')

    heatmap = compute_convolution(I, T)
    hm = plt.pcolor(heatmap)
    plt.colorbar(hm)
    plt.savefig('rl2_hm.jpg')
    (f_rows, f_cols, _) = T.shape
    output = predict_boxes(heatmap, (f_rows, f_cols), conf_thr=0.1)

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output
