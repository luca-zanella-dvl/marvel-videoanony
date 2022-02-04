from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path

from utils.general import increment_path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def extant_file(fname):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(fname):
        raise argparse.ArgumentTypeError(f"{fname} does not exist")
    return fname


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # object detection
        self.parser.add_argument(
            "--arch",
            default="yolo",
            help="model architecture. Currently only supports yolo",
        )

        # head detection
        self.parser.add_argument(
            "--head-model",
            type=extant_file,
            default=get_project_root() / 'weights/crowdhuman1280x_yolov5x6-xs.pt',
            help="head detection model.pt path",
        )
        self.parser.add_argument(
            "--head-imgsz",
            "--head-img",
            "--head-img-size",
            nargs="+",
            type=int,
            default=[1280],
            help="inference size h,w",
        )
        self.parser.add_argument(
            "--head-classes",
            nargs="+",
            type=int,
            help="filter by class: --classes 0, or --classes 0 2 3",
        )

        # vehicle detection
        self.parser.add_argument(
            "--veh-model",
            type=extant_file,
            default=get_project_root() / 'weights/yolov5x6.pt',
            help="vehicle detection model.pt path",
        )
        self.parser.add_argument(
            "--veh-imgsz",
            "--veh-img",
            "--veh-img-size",
            nargs="+",
            type=int,
            default=[1280],
            help="inference size h,w",
        )
        self.parser.add_argument(
            "--veh-classes",
            nargs="+",
            type=int,
            help="filter by class: --classes 0, or --classes 0 2 3",
        )
        
        self.parser.add_argument(
            "--lpd-model",
            type=extant_file,
            default=get_project_root() / 'weights/grn_ft_lpd_yolov5x.pt',
            help="license plate detection model.pt path",
        )
        self.parser.add_argument(
            "--lpd-imgsz",
            "--lpd-img",
            "--lpd-img-size",
            nargs="+",
            type=int,
            default=[640],
            help="inference size h,w",
        )
        self.parser.add_argument(
            "--lpd-classes",
            nargs="+",
            type=int,
            help="filter by class: --class 0, or --class 0 2 3",
        )

        # system
        self.parser.add_argument(
            "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="dataloader threads. 0 for single-thread.",
        )
        self.parser.add_argument(
            "--seed", type=int, default=317, help="random seed"
        )  # from CornerNet

        # IO
        self.parser.add_argument(
            "--source",
            required=True,
            help="file/dir/URL/glob",
        )
        self.parser.add_argument(
            "--project",
            default=get_project_root() / "runs/anonymize",
            help="save results to project/name",
        )
        self.parser.add_argument(
            "--name", default="exp", help="save results to project/name"
        )
        self.parser.add_argument(
            "--exist-ok",
            action="store_true",
            help="existing project/name ok, do not increment",
        )

    def parse(self, args=""):
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.head_imgsz *= 2 if len(opt.head_imgsz) == 1 else 1  # expand
        opt.veh_imgsz *= 2 if len(opt.veh_imgsz) == 1 else 1  # expand
        opt.lpd_imgsz *= 2 if len(opt.lpd_imgsz) == 1 else 1  # expand

        opt.save_dir = increment_path(
            Path(opt.project) / opt.name, exist_ok=opt.exist_ok
        )  # increment run
        opt.save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        print("The output will be saved to ", opt.save_dir)

        return opt

    def init(self, args=""):
        opt = self.parse(args)
        return opt
