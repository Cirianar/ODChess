# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Edited and adapted by Aidan Tiede 2021 for personal usage

import sys
import os
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from libs.window import Window

FILE = Path(__file__).resolve()

os.chdir(".")
ROOT = os.getcwd()
WEIGHTS = os.path.join(ROOT, "data\\models\\model_yolo5_v1\\checkpoint\\best.pt")
print(f"ROOT DIR: {ROOT}\nWEIGHTS: {WEIGHTS}")

# Imports from YOLOv5 module
sys.path.append("./modules/yolov5")
from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_imshow, check_requirements, non_max_suppression, print_args, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

class ObjectDetector:
    def __init__(self,
                weights,
                source,
                imgsz,
                conf_thres,
                iou_thres,
                max_det,
                device,
                view_img,
                classes,
                agnostic_nms,
                augment,
                line_thickness,
                hide_labels,
                hide_conf,
                half,
                ):
        self.weights = weights  # model.pt path(s)
        self.source = source  # file/dir/URL/glob, 0 for webcam
        self.imgsz = imgsz  # inference size (pixels)
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = max_det  # maximum detections per image
        self.device = device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = view_img  # show results
        self.classes = classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = agnostic_nms  # class-agnostic NMS
        self.augment = augment  # augmented inference
        self.line_thickness = line_thickness  # bounding box thickness (pixels)
        self.hide_labels = hide_labels  # hide labels
        self.hide_conf = hide_conf  # hide confidences
        self.half = half  # use FP16 half-precision inference

        self.attempt_load = attempt_load
        self.LoadStreams = LoadStreams
        self.check_imshow, self.check_requirements, self.non_max_suppression, self.print_args, self.scale_coords, self.set_logging = check_imshow, check_requirements, non_max_suppression, print_args, scale_coords, set_logging
        self.Annotator, self.colors = Annotator, colors
        self.select_device, self.time_sync = select_device, time_sync

        # Initalisation setup
        self.source = str(self.source) 
        self.weights = WEIGHTS

        self.set_logging()
        self.device = self.select_device(self.device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        self.infowindow = Window()

    def loadModel(self):
        stride = 64
        names = [f'class{i}' for i in range(1000)]  # assign defaults
        pt = ".pt"

        model = self.attempt_load(self.weights, map_location=self.device)
        stride = int(model.stride.max())  # model stride
        names = model.names  # get class names

        if self.half:
            model.half()  # to FP16
        
        return model, stride, names, pt 

    def loadStream(self, stride, pt):
        self.view_img = self.check_imshow()

        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = self.LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=pt)

        return dataset

    def scoreFrame(self, det, i, names, seen, im0s, img, t3, t2):
        seen += 1
        
        printString = f'{i}: '
        im0 = im0s[i].copy()

        printString += 'Input image dim(%gx%g), ' % img.shape[2:][::-1]  # adds the dimensions of the image to the print string
        annotator = self.Annotator(im0, line_width=self.line_thickness, example=str(names))

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for i in det[:, -1].unique():
                numDetections = (det[:, -1] == i).sum()  # detections per class
                printString += f"{numDetections} {names[int(i)]}{'s' * (numDetections > 1)}, "  # adds the number of detected objects and type of object to print string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Draws boxes on image
                if self.view_img:
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=self.colors(c, True))

        # Print time (inference-only)
        print(f'{printString}Done. ({t3 - t2:.3f}s)')

        # Stream results
        result = annotator.result()

        return seen, result

    def runInference(self, model, img, dt):
        t1 = self.time_sync()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        t2 = self.time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(img, augment=self.augment, visualize=False)[0]
        t3 = self.time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += self.time_sync() - t3

        return pred, t3, t2, dt
    
    def resizeResults(self, results, scale_percent):
        width = int(results.shape[1] * scale_percent)
        height = int(results.shape[0] * scale_percent)
        dim = (width, height)

        results = cv2.resize(results, dim)
        return results
            
    @torch.no_grad()
    def run(self):
        model, stride, names, pt = self.loadModel()
        dataset = self.loadStream(stride, pt)

        self.infowindow.buildWidgets(names)

        timeStamp = [0.0, 0.0, 0.0]

        for path, img, im0s, videoCapture, shape in dataset:
            predictions, time3, time2, detectionTime = self.runInference(model, img, timeStamp)
            
            for i, det in enumerate(predictions):  # per image
                numSeenObjects = 0
                numSeenObjects, imageResult = self.scoreFrame(det, i, names, numSeenObjects, im0s, img, time3, time2)
                
                imageResult = self.resizeResults(imageResult, 1.5)

                self.infowindow.updateWindow(det, names, imageResult)

                if 0xFF == ord('q'):
                    print(f"Quit program on '{0xFF}' click!")
                    break
                elif self.view_img:
                    cv2.imshow("ODChess", imageResult)
                    cv2.waitKey(1)  # 1 millisecond

        # Print results
        t = tuple(x / numSeenObjects * 1E3 for x in detectionTime)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

    def main(self, opt):
        self.print_args(FILE.stem, opt)

        self.check_requirements(exclude=('tensorboard', 'thop'))
        self.run()
