import os
import json
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from models.layers import compute_convolution
from models.matched_filtering import detect_red_light_mf


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Courses/2022Spring/CS148/RedLights2011_Medium'

# load splits: 
split_path = 'data/'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'preds/'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

# # DELETE
# filterM = np.load('preproc/night.npy')
# for img_name in ['RL-047.jpg']:
#     # read image using PIL:
#     I = Image.open(os.path.join(data_path,img_name))

#     # I = I.filter(ImageFilter.GaussianBlur(radius = 3))

#     # I.show()

#     # convert to numpy array:
#     I = np.asarray(I)
#     # I_mp = max_pool(I)
#     # I_ap = avg_pool(I)
#     # plt.imshow(I_mp)
#     # plt.savefig('MAXPOOL.jpg')
#     # plt.imshow(I_ap)
#     # plt.savefig('AVGPOOL.jpg')

#     heatmap = compute_convolution(I, filterM)
#     print(I.shape)
#     print(heatmap.shape)
#     plt.imshow(heatmap, cmap='hot')
#     hm = plt.pcolor(heatmap)
#     plt.colorbar(hm)
#     plt.savefig('rl2_hm_prime.jpg')
# assert False

# Make predictions on the training set.
preds_train = {}
for i in range(len(file_names_train)):
    if i % 25 == 0:
        print(f'File #{i}')

    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train_matched_filtering.json'),'w') as f:
    json.dump(preds_train, f)

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
