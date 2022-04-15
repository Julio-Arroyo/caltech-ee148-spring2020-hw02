import numpy as np


def matr_prod(m1, m2):
    if not np.shape(m1) == np.shape(m2):
        return 0

    m1_vec = m1.flatten()
    m2_vec = m2.flatten()

    # light invariance
    m1_vec = (1 / np.linalg.norm(m1_vec)) * m1_vec
    m2_vec = (1 / np.linalg.norm(m2_vec)) * m2_vec

    return np.inner(m1_vec, m2_vec)


def compute_convolution(I, T, stride=1):
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

    for i in range(0, hm_rows, stride):
        for j in range(0, hm_cols, stride):
            window = I[i: i + f_rows, j: j + f_cols]
            assert (np.shape(window) == np.shape(T),
                    f'Window, filter mismatch. (i,j)=({i, j}), window shape: {np.shape(window)}, I shape: {np.shape(I)}, T shape: {np.shape(T)}')

            window_vec = window.flatten()
            window_vec = (1 / np.linalg.norm(window_vec)) * window_vec
            heatmap[i, j] = np.inner(window_vec, filter_vec)

    return heatmap


def pool(I, pool_type, window_size=2, stride=1):
    (n_rows, n_cols, n_channels) = np.shape(I)

    # heatmap dimensions
    (out_rows, out_cols) = ((n_rows - window_size + 1) // stride,
                          (n_cols - window_size + 1) // stride)
    I_out = np.zeros((out_rows, out_cols, n_channels))

    for k in range(n_channels):
        for i in range(out_rows):
            for j in range(out_cols):
                window = I[i: i + window_size, j: j + window_size, k]
                if pool_type == 'avg':
                    I_out[i, j, k] = np.mean(window)
                elif pool_type == 'max':
                    I_out[i, j, k] = np.amax(window)
                else:
                    raise NotImplementedError
    return I_out


def avg_pool(I, window_size=2, stride=1):
    '''
    Average pooling layer
    '''
    return pool(I, 'avg', window_size=window_size, stride=stride)


def max_pool(I, window_size=2, stride=1):
    """Max pooling layer"""
    return pool(I, 'max', window_size=window_size, stride=stride)
