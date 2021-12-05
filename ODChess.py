# Chess piece object detection inference by Aidan Tiede using YOLOv5 & Google COLAB model and modules
"""
Run inference streams using a YOLOv5 .pt model.

Required dependencies:
    - Python3

Required modules
    - opencv-python
    - Torch
    - Pandas
    - Requests
    - pillow
    - Yq
    - Tqdm
    - Torchvision
    - Matplotlib
    - seaborn

Usage:
    $ python path/to/ODChess.py
"""
import argparse

from libs.detection import ObjectDetector

def buildObjectDetector(opt):
    return ObjectDetector(**vars(opt))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    chessOD = buildObjectDetector(opt)

    chessOD.main(opt)