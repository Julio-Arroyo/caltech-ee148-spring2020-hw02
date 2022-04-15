import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import json


if __name__ == '__main__':
    algo = 'ensemble'  # find_red, matched_filtering, ensemble
    # preds_path = f'./data/hw01_preds/{algo}'
    annotators = 'students_2021'  #students_2021, students_2020, mturk
    gts_path = f'data/ground_truth'
    data_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Courses/2022Spring/CS148/RedLights2011_Medium'

    # f = open(f'{preds_path}/preds.json')
    f = open(f'{gts_path}/formatted_annotations_{annotators}.json')
    img_2_bboxes = json.load(f)
    f_pred = open(f'preds/preds_train_{algo}.json')
    pred_bboxes = json.load(f_pred)
    fig, ax = plt.subplots()

    for img_name in img_2_bboxes:
        if img_name not in pred_bboxes:  # some images are saved for test set
            continue
        bboxes = img_2_bboxes[img_name]
        bboxes_pred = pred_bboxes[img_name]
        img = Image.open(f'{data_path}/{img_name}')
        ax.imshow(img)

        # GROUND TRUTHS
        for bbox in bboxes:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            
            # NOTE: this package uses x, y, width, height, rather than our
            # format TL and BR rows/cols
            rect = patches.Rectangle((bbox[1], bbox[0]),
                                     height,
                                     width,
                                     linewidth=2,
                                     edgecolor='m',
                                     facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

        # PREDICTIONS
        for bbox in bboxes_pred:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])

            # NOTE: this package uses x, y, width, height, rather than our
            # format TL and BR rows/cols
            rect_pred = patches.Rectangle((bbox[1], bbox[0]),
                                     height,
                                     width,
                                     linewidth=2,
                                     edgecolor='c',
                                     facecolor='none')
            ax.add_patch(rect_pred)
        img_id = img_name.strip('.jpg')
        # plt.savefig(f'{preds_path}/{img_id}_pred.jpg')
        plt.savefig(f'preds/images/{annotators}_{algo}/{img_id}_gt.jpg')

        plt.cla()
    f.close()
