import numpy as np


def get_bbox_area(bbox):
    if bbox[3] - bbox[1] <= 0 or bbox[2] - bbox[0] <= 0:
        return None
    area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    assert area > 0
    return area


def get_conf_score(algo, rgb_pred):
    if algo == 'find_red':
        weights = [0.5, 0.25, 0.25]  # weighting scheme for r,g,b z-scores

        rgb_vals = [(253,217,113), (240,67,97), (255,242,96), (253,168,113)]
        r_vals = [rgb[0] for rgb in rgb_vals]
        g_vals = [rgb[1] for rgb in rgb_vals]
        b_vals = [rgb[2] for rgb in rgb_vals]

        avg_r = sum(r_vals) / len(r_vals)
        std_r = np.std(r_vals)
        avg_g = sum(g_vals) / len(g_vals)
        std_g = np.std(g_vals)
        avg_b = sum(b_vals) / len(b_vals)
        std_b = np.std(b_vals)

        r_zscore = abs(rgb_pred[0] - avg_r) / std_r
        g_zscore = abs(rgb_pred[1] - avg_r) / std_r
        b_zscore = abs(rgb_pred[2] - avg_r) / std_r

        return weights[0] * r_zscore + weights[1] * g_zscore + weights[2] * b_zscore
    else:
        raise NotImplementedError("'{algo}'")
