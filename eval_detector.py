import os
import copy
import json
import numpy as np
from helpers import get_bbox_area


def compute_iou(box1, box2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # intersection box coords
    inter = [max(box1[0], box2[0]),  # row top left
             max(box1[1], box2[1]),  # col top left
             min(box1[2], box2[2]),  # row bottom right
             min(box1[3], box2[3])]  # col bottom right
    
    inter_area = get_bbox_area(inter)
    box1_area = get_bbox_area(box1)
    box2_area = get_bbox_area(box2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for pred_file, predictions in preds.iteritems():
        gt = copy.deepcopy(gts[pred_file])
        pred = copy.deepcopy(predictions)
        for i in range(len(gt)):
            found_match = False
            for j in range(len(predictions)):
                if pred[j] is None:
                    continue
                iou = compute_iou(pred[j][:4], gt[i])
                confidence = pred[j][4]
                if iou > iou_thr and confidence > conf_thr:
                    found_match = True
                    TP += 1
                    pred[j] = None
                    break
            if not found_match:
                FN += 1
        for curr_pred in pred:
            if curr_pred is not None:
                FP  += 1
    return TP, FP, FN


# set a path for predictions and annotations:
preds_path = '/preds'
gts_path = '/data/ground_truth'

# load splits:
split_path = '/data'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

# Load training predictions
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
