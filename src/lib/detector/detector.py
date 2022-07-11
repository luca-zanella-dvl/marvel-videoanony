import torch
import cv2
import numpy as np

from models.model import create_model, load_model
from datasets.dataset import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh


class Detector(object):
    def __init__(self, opt, model_path, imgsz, device):
        print("Creating model...")
        self.opt = opt
        self.model_path = model_path
        self.device = device
        self.model = create_model(opt.arch, model_path)
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(device)
        self.model.eval()
        self.model.float()

        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def pre_process(self, im0s):
        # Padded resize
        im = letterbox(im0s, self.imgsz, stride=self.stride)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    def post_process(self, pred, classes):
        # NMS
        pred = non_max_suppression(pred, classes=classes)
        return pred

    def process_im(self, im0, classes):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        im = self.pre_process(im0)

        # Inference
        pred = self.model(im)[0]
        pred = self.post_process(pred, classes)

        # results = []

        # Process predictions
        for i, det in enumerate(pred):  # per image                
            # gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        return pred

    def cls2label(self, cls):
        """
        For a given class, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        c = int(cls)  # integer class
        label = self.model.names[c]
        return label

