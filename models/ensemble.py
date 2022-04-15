from models.layers import matr_prod
import numpy as np


def load_filters():
    filters = []
    for i in range(4):
        filters.append(np.load(f'preproc/filter{i}.npy'))
    return filters
        

def find_best_filter(filters, I, i, j):
    max_confidence = float('-inf')
    filter_idx = -1
    for k in range(len(filters)):
        (f_rows, f_cols, _) = filters[k].shape
        window = I[i -4 : i + f_rows-4, j-4: j + f_cols-4]
        curr_confidence = matr_prod(window, filters[k])
        if curr_confidence > max_confidence:
            max_confidence = curr_confidence
            filter_idx = k
    return (max_confidence, filter_idx)


def find_red_matched_filtering(I, filters):
    '''
    4 random images used to find tones of red: {213, 291, 251, }
    '''
    # red_tones = {(253,217,113), (240,67,97), (255,242,96), (253,168,113), (230,108,133)}
    #               253,144,124, 
    # NOT INCLUDED
    # 227,41,82

    bboxes = []

    i = 0
    j = 0
    min_red = 235
    min_green = 65
    max_green = 245
    min_blue = 95
    max_blue = 115

    offset = 4
    
    # use while loops instead of for loops to jump around image, sorry i know it's ugly
    while i < 4 * (I.shape[0] // 5):  # look at top three-fifths of image only
        j = 0
        found_light = False
        while j < I.shape[1]:
            if (I[i, j, 0] > min_red
                and I[i, j, 1] > min_green and I[i, j, 1] < max_green
                and I[i, j, 2] > min_blue and I[i, j, 2] < max_blue):
                pred_confidence, filter_idx = find_best_filter(filters, I, i, j)
                bboxes.append([i - offset, j - offset, i + filters[filter_idx].shape[0] - offset,
                               j + filters[filter_idx].shape[1] - offset,
                               pred_confidence])
                found_light = True
            j += 3
        i += 3
    return bboxes