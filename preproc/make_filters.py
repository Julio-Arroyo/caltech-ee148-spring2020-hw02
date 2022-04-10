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
    lights = [(300, [
                    251.5,
                    261,
                    257.0,
                    265.5
                    ], 'tiny'),
            (162, [
                    187.66666666666669,
                    185,
                    220,  #216.33333333333331,
                    203.66666666666666
                    ], 'medium'),
            (10, [
                    179.5,
                    605.5,
                    235,  #198.5,
                    624.0
                    ], 'large'),
            (334, [
                    246.66666666666666,
                    91.66666666666667,
                    280,  # 268.0,
                    111.33333333333333
                    ], 'night')]

    for elem in lights:
        img_name = f'RL-{elem[0]:03}.jpg'
        img_name = f'{data_path}/{img_name}'
        img = Image.open(img_name)
        img = np.asarray(img)
        bbox = [int(elem[1][i]) for i in range(len(elem[1]))]
        filter_name = elem[2]
        save_filter(img, bbox, filter_name)
