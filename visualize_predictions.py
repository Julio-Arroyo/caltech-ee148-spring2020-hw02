import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import json


if __name__ == '__main__':
    algo = 'find_red'
    # preds_path = f'./data/hw01_preds/{algo}'
    annotators = 'mturk'  #students_2021, students_2020, mturk
    preds_path = f'data/ground_truth'
    data_path = '/Users/jarroyo/OneDrive - California Institute of Technology/Courses/2022Spring/CS148/RedLights2011_Medium'

    # f = open(f'{preds_path}/preds.json')
    f = open(f'{preds_path}/formatted_annotations_{annotators}.json')
    img_2_bboxes = json.load(f)

    fig, ax = plt.subplots()

    for img_name in img_2_bboxes:
        bboxes = img_2_bboxes[img_name]
        img = Image.open(f'{data_path}/{img_name}')
        ax.imshow(img)

        for bbox in bboxes:
            width = abs(bbox[2] - bbox[0])
            height = abs(bbox[3] - bbox[1])
            
            rect = patches.Rectangle((bbox[1], bbox[0]),
                                     width,
                                     height,
                                     linewidth=2,
                                     edgecolor='m',
                                     facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
        img_id = img_name.strip('.jpg')
        # plt.savefig(f'{preds_path}/{img_id}_pred.jpg')
        plt.savefig(f'{preds_path}/{annotators}/{img_id}_gt.jpg')

        plt.cla()
    f.close()
