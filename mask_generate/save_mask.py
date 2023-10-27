import os
import cv2
from table_det import table_det
import json


img_root = "/Users/haopengl1/Desktop/share/images"  # folder with images
img_list = os.listdir(img_root)
img_list = [p for p in img_list if not "2023" in p]

vis_mask = True  # whether to visualize the mask

if vis_mask:
    vis_root = "vis_table"
    os.makedirs(vis_root, exist_ok=True)

video_dict = {}
for res_name in img_list:
    video_id = res_name[:-8]
    if video_id not in video_dict:
        video_dict[video_id] = []
    video_dict[video_id].append(res_name)


result = {}

for video_id in video_dict:
    cut_id = video_id[-2:]
    previous_id = str(int(cut_id) - 1).zfill(2)
    latter_id = str(int(cut_id) + 1).zfill(2)
    video_id_preivous = video_id.replace(cut_id, previous_id)
    video_id_latter = video_id.replace(cut_id, latter_id)

    img_list_previous = video_dict.get(video_id_preivous, [])
    img_list_latter = video_dict.get(video_id_latter, [])

    all_img_list = []
    for img_name in img_list_previous + img_list_latter:
        img_path = os.path.join(img_root, img_name)
        img = cv2.imread(img_path)
        all_img_list.append(img)

    img_list = sorted(video_dict[video_id])

    for img_name in img_list:
        img_path = os.path.join(img_root, img_name)
        img = cv2.imread(img_path)
        all_img_list.append(img)

    table_corners = table_det(all_img_list)

    if isinstance(table_corners, str):
        print("skip {}".format(video_id), table_corners)
        continue

    table_corners, table_corners2 = table_corners
    result[video_id] = [table_corners.tolist(), table_corners2.tolist()]

    if vis_mask:
        img_ = img
        img_copy = img_.copy()
        cv2.fillPoly(img_, [table_corners], (255, 0, 0))
        alpha = 0.7  # Transparency factor.
        img = cv2.addWeighted(img_copy, alpha, img_, 1 - alpha, 0)
        cv2.imwrite(os.path.join(vis_root, img_name), img)

save_file = "table_corners.json"  # save four corners of the table

with open(save_file, "w") as f:
    json.dump(result, f, indent=4)
