def get_bbox_area(bbox):
    area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
    assert area > 0
    return area