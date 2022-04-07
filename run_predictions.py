import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality.
    '''
    (n_rows, n_cols, n_channels) = np.shape(I)
    (f_rows, f_cols, fchannels) = np.shape(T)  # filter dimensions

    assert n_channels == fchannels
    assert n_rows >= f_rows
    assert n_cols >= f_cols

    # heatmap dimensions
    (hm_rows, hm_cols) = ((n_rows - f_rows + 1) // stride,
                          (n_cols - f_cols + 1) // stride)
    heatmap = np.zeros((hm_rows, hm_cols))
    filter_vec = T.flatten()
    filter_vec = (1/np.linalg.norm(filter_vec)) * filter_vec  # light invariance

    for i in range(hm_rows):
        for j in range(hm_cols):
            window = I[i: i + f_rows, j: j + f_cols]
            assert (np.shape(window) == np.shape(T),
                    f'Window, filter mismatch. (i,j)=({i, j}), window shape: {np.shape(window)}, I shape: {np.shape(I)}, T shape: {np.shape(T)}')

            window_vec = window.flatten()
            window_vec = (1 / np.linalg.norm(window_vec)) * window_vec
            heatmap[i, j] = np.inner(window_vec, filter_vec)

    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    # IDEA 1: if local max (greater than all its neighbors), predict box around it
    # IDEA 2: hyperparameter threshold, predict box all around area above threshold
    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    box_height = 8
    box_width = 6

    num_boxes = np.random.randint(1,5)

    for i in range(num_boxes):
        (n_rows,n_cols,n_channels) = np.shape(I)

        tl_row = np.random.randint(n_rows - box_height)
        tl_col = np.random.randint(n_cols - box_width)
        br_row = tl_row + box_height
        br_col = tl_col + box_width

        score = np.random.random()

        output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''

    return output


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

    '''
    BEGIN YOUR CODE
    '''
    template_height = 8
    template_width = 6

    # You may use multiple stages and combine the results
    T = np.random.random((template_height, template_width))

    heatmap = compute_convolution(I, T)
    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Courses/2022Spring/CS148/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
