def get_center_of_bbox(bbox):
    """
    Get the center of a bounding box
    :param bbox: Bounding box coordinates (x1, y1, x2, y2)
    :return: Center coordinates (x, y)
    """
    x1, y1, x2, y2 = bbox

    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """
    Get the width of a bounding box
    :param bbox: Bounding box coordinates (x1, y1, x2, y2)
    :return: Width of the bounding box
    """
    x1, y1, x2, y2 = bbox

    return x2 - x1