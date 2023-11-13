# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
)
from utils.plots import Annotator, colors
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
import numpy as np
from utils.general import *
from motrackers import IOUTracker


card_name = [
    "JC",
    "B",
    "8H",
    "8S",
    "JS",
    "AS",
    "3C",
    "7C",
    "3D",
    "6S",
    "2H",
    "6C",
    "KH",
    "10D",
    "5H",
    "4S",
    "7D",
    "5S",
    "QS",
    "KD",
    "3H",
    "2D",
    "2C",
    "5C",
    "QC",
    "4D",
    "AH",
    "KS",
    "2S",
    "U",
    "7H",
    "AC",
    "8C",
    "5D",
    "9S",
    "10H",
    "AD",
    "8D",
    "4C",
    "3S",
    "KC",
    "4H",
    "7S",
    "QH",
    "QD",
    "JH",
    "6H",
    "6D",
    "9H",
    "9C",
    "JD",
    "10S",
    "9D",
    "10C",
]


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob, 0 for webcam
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-seg",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
):
    tracker = IOUTracker(iou_threshold=0.8)

    res_root = source + "_res"
    res_files = sorted(os.listdir(res_root))

    id_box = {}
    id_list = []
    cls_list = []

    for i, res_file in enumerate(res_files):
        res_file_path = os.path.join(res_root, res_file)
        with open(res_file_path, "r") as f:
            lines = f.readlines()
        detection_class_ids = []
        detection_boxes = []
        detection_confidences = []
        cls_temp = []
        for line in lines:
            cls, x1, y1, x2, y2, conf = [eval(x) for x in line.strip().split(" ")[:6]]
            detection_class_ids.append(cls)
            detection_boxes.append([x1, y1, x2, y2])
            detection_confidences.append(conf)
        detection_class_ids = np.array(detection_class_ids)
        boxes = np.array(detection_boxes)
        boxes_ = boxes.copy()
        boxes_[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_[:, 1] = boxes[:, 1] - boxes[:, 3] / 2

        id_temp = []

        output_tracks = tracker.update(
            boxes_, detection_confidences, detection_class_ids
        )
        for track in output_tracks:
            (
                frame,
                id,
                bb_left,
                bb_top,
                bb_width,
                bb_height,
                confidence,
                x,
                y,
                z,
                c,
            ) = track
            if not id in id_box:
                id_box[id] = []
            id_box[id].append(
                [
                    bb_left,
                    bb_top,
                    bb_width,
                    bb_height,
                ]
            )
            id_temp.append(id)
            cls_temp.append(c)
        id_list.append(id_temp)
        cls_list.append(cls_temp)

    for id in id_box:
        l = len(id_box[id])
        start = min(3, l - 1)
        id_box[id] = np.array(id_box[id][start:]).mean(axis=0)

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run

    txt_root = os.path.join(save_dir, "angle_res")
    os.makedirs(txt_root, exist_ok=True)
    vis_root = os.path.join(save_dir, "vis")
    os.makedirs(vis_root, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    times = 0
    cnt = 0
    id2angle = {}
    for path, im, im0s, vid_cap, s in dataset:
        im0s_ = cv2.GaussianBlur(im0s, (11, 11), 5)
        if times == 0:
            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize = (
                    increment_path(save_dir / Path(path).stem, mkdir=True)
                    if visualize
                    else False
                )
                pred, out = model(im, augment=augment, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(
                    pred,
                    conf_thres,
                    iou_thres,
                    classes,
                    agnostic_nms,
                    max_det=max_det,
                    nm=32,
                )

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f"{i}: "
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                p = Path(p)  # to Path

                s += "%gx%g " % im.shape[2:]  # print string
                if not len(det) == 1:
                    continue
                if len(det):
                    masks = process_mask(
                        proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True
                    )  # HWC

                    mask_ = masks.permute(1, 2, 0).cpu().numpy()
                    mask_ = scale_masks(im.shape[2:], mask_, im0.shape)
                    mask_ = mask_.astype(np.uint8) * 255
                    edges = cv2.Canny(mask_, 150, 300)

                    # Apply HoughLinesP method to
                    # to directly obtain line end points
                    lines = cv2.HoughLinesP(
                        edges,  # Input edge image
                        1,  # Distance resolution in pixels
                        np.pi / 180,  # Angle resolution in radians
                        threshold=30,  # Min number of votes for valid line
                        minLineLength=100,  # Min allowed length of line
                        maxLineGap=100,  # Max allowed gap between line for joining them
                    )

                    line_45 = []
                    line_135 = []

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

                    line_45 = process_line_group(line_45)
                    line_135 = process_line_group(line_135)

                    lines_list = (
                        line_45
                        + line_135
                        + [[0, im0s.shape[0] - 1, 1, im0s.shape[0] - 1]]
                    )

                    intersection_list = []
                    intersection_list2 = []
                    for i in range(len(lines_list)):
                        for j in range(i + 1, len(lines_list)):
                            line1 = np.array(lines_list[i]).reshape(2, 2)
                            line2 = np.array(lines_list[j]).reshape(2, 2)

                            xdiff = (
                                line1[0][0] - line1[1][0],
                                line2[0][0] - line2[1][0],
                            )
                            ydiff = (
                                line1[0][1] - line1[1][1],
                                line2[0][1] - line2[1][1],
                            )

                            div = det_(xdiff, ydiff)
                            if div == 0:
                                continue
                            d = (det_(*line1), det_(*line2))
                            x = det_(d, xdiff) / div
                            y = det_(d, ydiff) / div

                            if i != len(lines_list) - 1 and j != len(lines_list) - 1:
                                if 0 <= x <= im0s.shape[1] and 0 <= y:
                                    intersection_list2.append((int(x), int(y)))

                            if (
                                x <= 0
                                or y <= 0
                                or x >= im0s.shape[1]
                                or y >= im0s.shape[0]
                            ):
                                continue
                            if y == im0s.shape[0] - 1:
                                if x < im0s.shape[1] / 5 or x > im0s.shape[1] / 5 * 4:
                                    continue

                            intersection_list.append((int(x), int(y)))

                    if len(intersection_list) == 6:
                        intersection_list = [
                            p
                            for p in intersection_list
                            if not p[1] == im0s.shape[0] - 1
                        ]

                    if len(intersection_list) != 4 and len(intersection_list) != 5:
                        continue

                    # if not len(intersection_list2) == 4:
                    #     breakpoint()

                    # intersection_list_np = sort_points(intersection_list)
                    intersection_list_np2 = sort_points(intersection_list2)
                    times += 1

        label_path = (
            path.replace(source, source + "_res")
            .replace(".jpg", ".txt")
            .replace(".png", ".txt")
        )
        if not os.path.exists(label_path):
            continue

        objs = id_list[cnt]

        edges = cv2.Canny(im0s_, 100, 150)
        W, H = im0s.shape[1], im0s.shape[0]
        line_list = []

        error_obj = []

        for i, obj in enumerate(objs):
            x, y, w, h = id_box[obj].tolist()
            xmin = int(x * W)
            ymin = int(y * H)
            xmax = int((x + w) * W)
            ymax = int((y + h) * H)
            cropped_edges = edges[ymin:ymax, xmin:xmax]

            lines = cv2.HoughLinesP(
                cropped_edges,  # Input edge image
                1,  # Distance resolution in pixels
                np.pi / 180,  # Angle resolution in radians
                threshold=20,  # Min number of votes for valid line
                minLineLength=5,  # Min allowed length of line
                maxLineGap=10,  # Max allowed gap between line for joining them
            )
            if lines is not None:
                length_list = []
                for points in lines:
                    x1, y1, x2, y2 = points[0]
                    length = (x1 - x2) ** 2 + (y1 - y2) ** 2
                    length_list.append(length)
                top1_line = lines[np.argmax(length_list)][0]
                line_list.append(top1_line + np.array([xmin, ymin, xmin, ymin]))
            else:
                error_obj.append(i)
                line_list.append(np.zeros(4))

        if len(line_list) == 0:
            continue

        point_list = []
        for l in line_list:
            x1, y1, x2, y2 = l
            # cv2.line(im0s, l[:2], l[2:], (0, 0, 255), 2, cv2.LINE_AA)
            point_list.append(l[:2])
            point_list.append(l[2:])
        point_list = np.float32(point_list)
        trans_matrix = cv2.getPerspectiveTransform(
            intersection_list_np2.astype(np.float32), pts2
        )

        point_list_t = (
            cv2.perspectiveTransform(point_list[None, :, :], trans_matrix)[0]
            .reshape(-1, 2, 2)
            .astype(np.int32)
        )

        # new_pic = np.ones((S, S, 3), dtype=np.uint8) * 200
        player_dict = {i: [] for i in range(4)}
        for i, line in enumerate(point_list_t):
            if i in error_obj:
                continue
            x1, y1 = line[0]
            x2, y2 = line[1]
            center = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]]).repeat(4, axis=0)
            dist = np.sum((center - player_center) ** 2, axis=1)
            belong_to = np.argmin(dist)
            player_dict[belong_to].append((line, i))
        res_dict = {}
        for i in range(4):
            line_list = player_dict[i]
            random_color = hex[i]
            for line, t in line_list:
                line_ = to_arrow(line, i, S)
                line_ = np.array(line_)
                cos = np.dot(line_[1] - line_[0], player_arrow[i]) / (
                    np.linalg.norm(line_[1] - line_[0])
                    * np.linalg.norm(player_arrow[i])
                )
                angle = str(round(np.arccos(cos) * 180 / np.pi % 180))

                res_dict[t] = (angle, i)

        for i, obj in enumerate(objs):
            x, y, w, h = id_box[obj].tolist()
            xmin = int(x * W)
            ymin = int(y * H)
            xmax = int((x + w) * W)
            ymax = int((y + h) * H)
            if obj in id2angle:
                angle, t = id2angle[obj]
            else:
                if i in res_dict:
                    angle, t = res_dict[i]
                else:
                    angle = "?"
                    t = 4
                id2angle[obj] = (angle, t)
            random_color = hex[t]
            # cv2.rectangle(im0s, (xmin, ymin), (xmax, ymax), random_color, 2)
            # cv2.putText(
            #     im0s,
            #     cls_name,
            #     (xmin, ymin - 4),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.8,
            #     random_color,
            #     2,
            #     cv2.LINE_AA,
            # )
            p1, p2 = (xmin, ymin), (xmax, ymax)
            cv2.rectangle(im0s, p1, p2, random_color, thickness=2, lineType=cv2.LINE_AA)
            lw = 3
            tf = max(lw - 1, 1)
            cls_name = card_name[cls_list[cnt][i]]
            cls_name = " ".join([cls_name, angle])
            w, h = cv2.getTextSize(cls_name, 0, fontScale=lw / 3, thickness=tf)[0]
            p2 = p1[0] + w, p1[1] - h - 3
            cv2.rectangle(im0s, p1, p2, random_color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                im0s,
                cls_name,
                (p1[0], p1[1] - 2),
                0,
                lw / 3,
                (255, 255, 255),
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

            # cv2.putText(
            #     im0s,
            #     id2player[t] + " " + angle,
            #     (xmin, ymax + 20),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.8,
            #     random_color,
            #     2,
            #     cv2.LINE_AA,
            # )

        out_data = []
        for i in range(len(objs)):
            if i in res_dict:
                angle, t = res_dict[i]
            else:
                angle, t = "?", 4
            out_data.append((angle, t))

        save_path = (
            os.path.join(txt_root, Path(path).name)
            .replace(".jpg", ".txt")
            .replace(".png", ".txt")
        )
        with open(save_path, "w") as f:
            for angle, t in out_data:
                f.write(f"{angle} {id2player[t]}\n")
        print(label_path)
        cv2.imwrite(os.path.join(vis_root, Path(path).name), im0s)
        cnt += 1


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s-seg.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project",
        default=ROOT / "runs/predict-seg",
        help="save results to project/name",
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=3, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
