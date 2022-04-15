import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from helpers import get_bbox_area

KEY_ERR = set()

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
    if inter_area is None:
        return 0
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

    for pred_file, predictions in preds.items():
        # pre-select only the confident predictions
        confident_preds = []
        for curr_pred in predictions:
            if curr_pred[4] > conf_thr:
                confident_preds.append(curr_pred)

        gts_set = set()
        for gt in gts[pred_file]:
            gts_set.add(tuple(gt[:4]))  # add each bounding box
        preds_set = set()
        for curr_pred in confident_preds:
            preds_set.add(tuple(curr_pred[:4]))  # add each prediction

        for curr_pred in confident_preds:
            # for each prediction, try to find a ground truth to match it to
            for curr_gt in gts[pred_file]:
                iou = compute_iou(curr_pred[:4], curr_gt[:4])
                if iou > iou_thr:
                    TP += 1
                    try:
                        gts_set.remove(tuple(curr_gt[:4]))
                    except KeyError:
                        KEY_ERR.add(tuple(curr_gt[:4]))
                    preds_set.remove(tuple(curr_pred[:4]))
                    break
        FP += len(preds_set)  # if a prediction did not have a ground truth to match to, it is a false positive
        FN += len(gts_set)  # if no prediction was found to match a ground truth, that gt is a false negative
    return TP, FP, FN


# set a path for predictions and annotations:
preds_path = 'preds/'
gts_path = 'data/ground_truth'

# load splits:
split_path = 'data/'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

algo = 'matched_filtering'  # 'ensemble', 'naive', 'matched_filtering'

# Load training predictions
with open(os.path.join(preds_path, f'preds_train_{algo}.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'formatted_annotations_students_2021.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path, f'preds_test_{algo}.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'formatted_annotations_students_2021.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 
plt.xlim([0,0.25])
plt.ylim([0,1])
confidence_thrs = sorted([pred[4] for fname in preds_train for pred in preds_train[fname]]) # using (ascending) list of confidence scores as thresholds
for iou_thr in [0.25, 0.5, 0.75]:
    tp_train = np.zeros((len(confidence_thrs),))
    fp_train = np.zeros((len(confidence_thrs),))
    fn_train = np.zeros((len(confidence_thrs),))

    print('WATCH OUT')
    print(len(confidence_thrs))

    # tp_train, fp_train, fn_train = compute_counts(preds_train, gts_train)
    # print(f'TP: {tp_train}. FP: {fp_train}. FN: {fn_train}.')
    # print(f'Precision: {tp_train / (tp_train + fp_train)}. Recall: {tp_train / (tp_train + fn_train)}')
    for i in range(len(confidence_thrs)):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=confidence_thrs[i])

    # Plot training set PR curves
    precisions = np.zeros((len(confidence_thrs),))
    recalls = np.zeros((len(confidence_thrs),))
    for i in range(tp_train.shape[0]):
        print(tp_train[i])
        print(fp_train[i])
        print(fn_train[i])
        precisions[i] = tp_train[i] / (tp_train[i] + fp_train[i])
        recalls[i] = tp_train[i] / (tp_train[i] + fn_train[i])
    plt.plot(recalls, precisions)
plt.legend(['IOU THR: 0.25', 'IOU THR: 0.5', 'IOU THR: 0.75'])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig(f'{algo}_VARYING_CONFIDENCE_labeled.jpg')

plt.clf()
if done_tweaking:
    confidence_thrs_test = sorted([pred[4] for fname in preds_test for pred in preds_test[fname]]) # using (ascending) list of confidence scores as thresholds
    for iou_thr in [0.25, 0.5, 0.75]:
        tp_test = np.zeros((len(confidence_thrs_test),))
        fp_test = np.zeros((len(confidence_thrs_test),))
        fn_test = np.zeros((len(confidence_thrs_test),))

        # tp_train, fp_train, fn_train = compute_counts(preds_test, gts_train)
        # print(f'TP: {tp_train}. FP: {fp_train}. FN: {fn_train}.')
        # print(f'Precision: {tp_train / (tp_train + fp_train)}. Recall: {tp_train / (tp_train + fn_train)}')
        for i in range(len(confidence_thrs_test)):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr, conf_thr=confidence_thrs_test[i])

        # Plot testing set PR curves
        precisions_test = np.zeros((len(confidence_thrs_test),))
        recalls_test = np.zeros((len(confidence_thrs_test),))
        for i in range(tp_test.shape[0]):
            precisions_test[i] = tp_test[i] / (tp_test[i] + fp_test[i])
            recalls_test[i] = tp_test[i] / (tp_test[i] + fn_test[i])
        plt.plot(list(recalls_test), precisions_test)
    plt.legend(['IOU THR: 0.25', 'IOU THR: 0.5', 'IOU THR: 0.75'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'{algo}_VARYING_CONFIDENCE_TEST2.jpg')
