from PIL import Image
import numpy as np
import json


# to get filter: images 10, 308, 331, 307, 300, 299, 172, 162, 158
# types: far, normal, close, night


data_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Courses/2022Spring/CS148/RedLights2011_Medium'

def save_filter(img, bbox, fname):
    """
    Given an image (numpy array) and a bounding box of the region intended to be a
    filter, crop it and save it as numpy array
    """
    filter = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    with open(f'{fname}.npy', 'wb') as f:
        np.save(f, filter)
    return


if __name__ == '__main__':
    lights = [(13, [
                    154.33333333333331,
                    325.3333333333333,
                    160.33333333333331,
                    331.3333333333333
                    ]),
            (120, [
                    231,
                    417.6666666666667,
                    245.33333333333334,
                    431.6666666666667
                    ]),
            (10, [
                    31,
                    326,
                    50,
                    346
                    ]),
            (247, [
                    182.33333333333331,
                    494,
                    207.66666666666669,
                    515.6666666666666
                    ]),
            (25, [
                    152.33333333333331,
                    326.6666666666667,
                    159.33333333333331,
                    333.0
                    ]),
            (57, [
                    224.33333333333334,
                    316.6666666666667,
                    230.33333333333334,
                    323.6666666666667
                    ])]

    for idx, elem in enumerate(lights):
        img_name = f'RL-{elem[0]:03}.jpg'
        img_name = f'{data_path}/{img_name}'
        img = Image.open(img_name)
        img = np.asarray(img)
        bbox = [int(elem[1][i]) for i in range(len(elem[1]))]
        save_filter(img, bbox, f'preproc/filter{idx}')
