from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path

import _init_paths

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm

from lib.detector.detector import Detector
from lib.datasets.dataset import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from lib.opts import opts

from lib.utils.torch_utils import select_device


HEAD_LABEL = 'head'
VEHICLES_LABELS = ['car', 'motorcycle', 'truck']
LICENSE_PLATE_LABEL = 'license plate'


@torch.no_grad()
def main(opt):
    is_file = Path(opt.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = opt.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or (is_url and not is_file)

    # Load models
    device = select_device(opt.device)
    head_detector = Detector(opt, opt.head_model, opt.head_imgsz, device)
    veh_detector = Detector(opt, opt.veh_model, opt.veh_imgsz, device)
    lp_detector = Detector(opt, opt.lpd_model, opt.lpd_imgsz, device)

    head_classes = head_detector.model.names.index(HEAD_LABEL)
    veh_classes = [veh_detector.model.names.index(veh) for veh in VEHICLES_LABELS]
    lp_classes = lp_detector.model.names.index(LICENSE_PLATE_LABEL)

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source)
        bs = len(dataset)
    else:
        dataset = LoadImages(opt.source)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    with tqdm(total=len(dataset)) as pbar:
        # Run inference
        for path, im0s, vid_cap, s in dataset:

            for i in range(bs):
                to_anonymize = []

                if webcam:
                    p, im0, = path[i], im0s[i].copy()
                else:
                    p, im0, = path, im0s.copy()

                pbar.set_description("Anonymizing %s" % p)

                # Inference
                p = Path(p)  # to Path
                save_path = str(opt.save_dir / p.name)  # im.jpg

                # Head detection
                head_labels = head_detector.process_im(im0, head_classes)
                for head_label, *head_xyxy, head_conf in head_labels:
                    head_bbox = torch.tensor(head_xyxy).view(1, 4).clone().view(-1).numpy().astype(np.int32)
                    to_anonymize.append((head_bbox[1], head_bbox[3], head_bbox[0], head_bbox[2]))

                # Vehicle detection
                veh_labels = veh_detector.process_im(im0, veh_classes)
                for veh_label, *veh_xyxy, veh_conf in veh_labels:
                    veh_bbox = torch.tensor(veh_xyxy).view(1, 4).clone().view(-1).numpy().astype(np.int32)
                    veh_im0 = im0[veh_bbox[1] : veh_bbox[3], veh_bbox[0] : veh_bbox[2]]

                    # License plate detection
                    lp_labels = lp_detector.process_im(veh_im0, lp_classes)
                    for lp_label, *lp_xyxy, lp_conf in lp_labels:
                        lp_bbox = (torch.tensor(lp_xyxy).view(1, 4)).view(-1).numpy().astype(np.int32)
                        to_anonymize.append((veh_bbox[1] + lp_bbox[1], veh_bbox[1] + lp_bbox[3], veh_bbox[0] + lp_bbox[0], veh_bbox[0] + lp_bbox[2]))

                for miny, maxy, minx, maxx in to_anonymize:
                        sub_im = im0[miny : maxy, minx : maxx]
                        sub_im = cv2.GaussianBlur(sub_im, (45, 45), 30)
                        im0[
                            miny : maxy,
                            minx : maxx,
                        ] = sub_im

                im0 = np.asarray(im0)
                # Save results (anonymised image)
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = dataset.fps[i], im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                pbar.update(1)


if __name__ == "__main__":
    opt = opts().init()
    main(opt)
