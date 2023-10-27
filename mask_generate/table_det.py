import cv2
import numpy as np
from skimage.morphology import (
    erosion,
    dilation,
    opening,
    area_closing,
)

from skimage.measure import label, regionprops


square = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def to_arrow(line, pos, S):
    x1, y1 = line[0]
    x2, y2 = line[1]
    if pos % 2 == 0:
        y1_dist = abs(y1 - S / 2)
        y2_dist = abs(y2 - S / 2)
        if y1_dist < y2_dist:
            x1, y1, x2, y2 = x2, y2, x1, y1
        elif y1_dist == y2_dist:
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
    else:
        x1_dist = abs(x1 - S / 2)
        x2_dist = abs(x2 - S / 2)
        if x1_dist < x2_dist:
            x1, y1, x2, y2 = x2, y2, x1, y1
        elif x1_dist == x2_dist:
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
    return [[x1, y1], [x2, y2]]


def sort_points(intersection_list):
    intersection_list_np = np.array(intersection_list)
    center = np.mean(intersection_list_np, axis=0)
    siftted_points = intersection_list_np - center

    angle = np.arctan2(siftted_points[:, 1], siftted_points[:, 0]) * 180 / np.pi

    angle[angle < 0] = angle[angle < 0] + 360

    sorted_index = np.argsort(angle)

    intersection_list_np = intersection_list_np[sorted_index]

    y = intersection_list_np[:, 1]
    y_min_index = np.argmin(y)
    final = np.concatenate(
        [intersection_list_np[y_min_index:], intersection_list_np[:y_min_index]], axis=0
    )

    return final


def calc_area(line1, line2):
    line1 = np.array(line1).reshape(2, 2)
    line2 = np.array(line2).reshape(2, 2)
    point_list = np.concatenate([line1, line2], axis=0)
    center = np.mean(point_list, axis=0)
    angle = (
        np.arctan2(point_list[:, 1] - center[1], point_list[:, 0] - center[0])
        * 180
        / np.pi
    )
    angle[angle < 0] = angle[angle < 0] + 360
    sorted_index = np.argsort(angle)
    point_list = point_list[sorted_index]
    # calculate area given four points
    x = point_list[:, 0]
    y = point_list[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area


def multi_dil(im, num, element=square):
    for i in range(num):
        im = dilation(im, element)
    return im


def multi_ero(im, num, element=square):
    for i in range(num):
        im = erosion(im, element)
    return im


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def process_line_group(line_list):
    assert len(line_list) >= 2, "line_list should have at least 2 lines"
    if len(line_list) == 2:
        return line_list
    else:
        id_list = []
        area_list = []
        for i in range(len(line_list)):
            for j in range(i + 1, len(line_list)):
                id_list.append((i, j))
                area_list.append(
                    calc_area(
                        line_list[i],
                        line_list[j],
                    )
                )
        top1_id = np.argsort(area_list)[-1]
        line1 = [line_list[id_list[top1_id][0]][p] for p in range(4)]
        line2 = [line_list[id_list[top1_id][1]][p] for p in range(4)]

        return [line1, line2]


def table_det(all_img_list):
    mask_final = np.zeros(all_img_list[0].shape[:2]) > 1

    for image in all_img_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask = cv2.inRange(
            image, (50, 125, 50), (80, 255, 255)
        )  # range of green color (table) in HSV
        mask = mask > 0
        mask_final = np.logical_or(mask_final, mask)

    mask = mask_final

    mask = mask.astype(np.uint8) * 255

    mask = multi_dil(mask, 3)
    mask = area_closing(mask, 2500)
    mask = multi_ero(mask, 1)

    open = opening(mask)
    label_im = label(open)
    regions = regionprops(label_im)

    area_list = [p.area for p in regions]
    top_2_id = np.argsort(area_list)[-2:]

    if (label_im == (top_2_id[0] + 1)).sum() * 3 < (
        label_im == (top_2_id[1] + 1)
    ).sum():
        mask = label_im == (top_2_id[1] + 1)
    else:
        mask = (label_im == (top_2_id[0] + 1)) + (label_im == (top_2_id[1] + 1))

    mask = mask.astype(np.uint8) * 255

    # Use canny edge detection
    edges = cv2.Canny(mask, 150, 300)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=60,  # Min number of votes for valid line
        minLineLength=50,  # Min allowed length of line
        maxLineGap=100,  # Max allowed gap between line for joining them
    )

    line_45 = []
    line_135 = []
    if lines is None:
        return "error0"

    for points in lines:
        x1, y1, x2, y2 = points[0]
        alpha = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi % 180
        if 15 <= alpha <= 75:
            new_line = np.array([x1, y1]) + [
                np.cos(alpha * np.pi / 180) * 1000,
                np.sin(alpha * np.pi / 180) * 1000,
            ]
            new_line = np.concatenate([np.array([x1, y1]), new_line])
            line_45.append(new_line)
        elif 105 <= alpha <= 165:
            new_line = np.array([x1, y1]) + [
                np.cos(alpha * np.pi / 180) * 1000,
                np.sin(alpha * np.pi / 180) * 1000,
            ]
            new_line = np.concatenate([np.array([x1, y1]), new_line])
            line_135.append(new_line)

    if len(line_45) < 2 or len(line_135) < 2:
        return "error1"

    line_45 = process_line_group(line_45)
    line_135 = process_line_group(line_135)

    lines_list = line_45 + line_135 + [[0, image.shape[0] - 1, 1, image.shape[0] - 1]]

    intersection_list = []
    intersection_list2 = []
    for i in range(len(lines_list)):
        for j in range(i + 1, len(lines_list)):
            line1 = np.array(lines_list[i]).reshape(2, 2)
            line2 = np.array(lines_list[j]).reshape(2, 2)

            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            div = det(xdiff, ydiff)
            if div == 0:
                continue
            d = (det(*line1), det(*line2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div

            if i != len(lines_list) - 1 and j != len(lines_list) - 1:
                if 0 <= x <= image.shape[1] and 0 <= y:
                    intersection_list2.append((int(x), int(y)))

            if x <= 0 or y <= 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            if y == image.shape[0] - 1:
                if x < image.shape[1] / 5 or x > image.shape[1] / 5 * 4:
                    continue

            intersection_list.append((int(x), int(y)))

    if len(intersection_list) == 6:
        intersection_list = [
            p for p in intersection_list if not p[1] == image.shape[0] - 1
        ]

    if len(intersection_list) != 4 and len(intersection_list) != 5:
        return "error2"

    intersection_list_np = sort_points(intersection_list)
    intersection_list_np2 = sort_points(intersection_list2)

    return intersection_list_np, intersection_list_np2
