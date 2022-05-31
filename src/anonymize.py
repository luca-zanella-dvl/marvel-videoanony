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


HEAD_LABEL = "head"
VEHICLES_LABELS = ["car", "motorcycle", "bus", "truck"]
LICENSE_PLATE_LABEL = "license plate"


@torch.no_grad()
def main(opt):
    is_file = Path(opt.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = opt.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )
    webcam = (
        opt.source.isnumeric()
        or opt.source.endswith(".txt")
        or (is_url and not is_file)
    )

    # Load models
    device = select_device(opt.device)

    head_detector = Detector(opt, opt.head_model, opt.head_imgsz, device)
    head_classes = head_detector.model.names.index(HEAD_LABEL)

    if opt.two_stage:
        veh_detector = Detector(opt, opt.veh_model, opt.veh_imgsz, device)
        veh_classes = [veh_detector.model.names.index(veh) for veh in VEHICLES_LABELS]

    lp_detector = Detector(opt, opt.lpd_model, opt.lpd_imgsz, device)
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

    frame = 0
    with tqdm(total=len(dataset)) as pbar:
        # Run inference
        for path, im0s, vid_cap, s in dataset:
            for i in range(bs):
                to_anonymize = []

                if webcam:
                    p, im0, = (
                        path[i],
                        im0s[i].copy(),
                    )
                else:
                    p, im0, = (
                        path,
                        im0s.copy(),
                    )

                pbar.set_description("Anonymizing %s" % p)

                # Inference
                p = Path(p)  # to Path
                save_path = str(opt.save_dir / p.name)  # im.jpg

                imc = im0.copy()
                # Head detection
                head_labels = head_detector.process_im(imc, head_classes)
                for head_label, *head_xyxy, head_conf in head_labels:
                    head_bbox = (
                        torch.tensor(head_xyxy)
                        .view(1, 4)
                        .clone()
                        .view(-1)
                        .numpy()
                        .astype(np.int32)
                    )
                    # miny, maxy, minx, maxx
                    to_anonymize.append(
                        (head_bbox[1], head_bbox[3], head_bbox[0], head_bbox[2])
                    )

                imc = im0.copy()
                # Vehicle detection
                if opt.two_stage:
                    veh_labels = veh_detector.process_im(imc, veh_classes)
                    for veh_label, *veh_xyxy, veh_conf in veh_labels:
                        veh_bbox = (
                            torch.tensor(veh_xyxy)
                            .view(1, 4)
                            .clone()
                            .view(-1)
                            .numpy()
                            .astype(np.int32)
                        )
                        veh_im0 = im0[
                            veh_bbox[1] : veh_bbox[3], veh_bbox[0] : veh_bbox[2]
                        ]
                        veh_imc = veh_im0.copy()

                        # License plate detection
                        lp_labels = lp_detector.process_im(veh_imc, lp_classes)
                        for lp_label, *lp_xyxy, lp_conf in lp_labels:
                            lp_bbox = (
                                (torch.tensor(lp_xyxy).view(1, 4))
                                .view(-1)
                                .numpy()
                                .astype(np.int32)
                            )
                            # miny, maxy, minx, maxx
                            to_anonymize.append(
                                (
                                    veh_bbox[1] + lp_bbox[1],
                                    veh_bbox[1] + lp_bbox[3],
                                    veh_bbox[0] + lp_bbox[0],
                                    veh_bbox[0] + lp_bbox[2],
                                )
                            )
                else:
                    # License plate detection
                    lp_labels = lp_detector.process_im(imc, lp_classes)
                    for lp_label, *lp_xyxy, lp_conf in lp_labels:
                        lp_bbox = (
                            (torch.tensor(lp_xyxy).view(1, 4))
                            .view(-1)
                            .numpy()
                            .astype(np.int32)
                        )
                        # miny, maxy, minx, maxx
                        to_anonymize.append(
                            (lp_bbox[1], lp_bbox[3], lp_bbox[0], lp_bbox[2])
                        )

                for miny, maxy, minx, maxx in to_anonymize:
                    sub_im = im0[miny:(maxy+1), minx:(maxx+1)]
                    sub_im = cv2.GaussianBlur(sub_im, (45, 45), 30)
                    im0[
                        miny:(maxy+1),
                        minx:(maxx+1),
                    ] = sub_im

                # im0 = np.asarray(im0)

                # Save results (anonymised image)
                if dataset.mode == "image":
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
                            fps = opt.stream_fps if opt.stream_fps is not None else dataset.fps[i]
                            w, h = im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        # vid_writer[i] = cv2.VideoWriter(
                        #     save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                        # )
                        vid_writer[i] = cv2.VideoWriter(gstreamer_pipeline_out(opt.stream_uri), cv2.CAP_GSTREAMER, 0, fps, (w, h), True)
                    if not vid_writer[i].isOpened():
                        raise Exception("can't open video writer")
                    vid_writer[i].write(im0)
                    print("frame written to the server")

                pbar.update(1)


def gstreamer_pipeline_out(stream_uri):
    # return (
    #     'appsrc ! videoconvert' + \
    #     ' ! x264enc speed-preset=ultrafast tune=zerolatency' + \
    #     ' ! rtspclientsink protocols=tcp location=rtsp://172.17.0.3:8554/mystream'
    # )
    # return (
    #     'appsrc name=appsrc format=time is-live=true caps=video/x-raw,format=(string)BGR appsrc. ! videoconvert' + \
    #     ' ! x264enc tune=zerolatency' + \
    #     ' ! rtspclientsink protocols=tcp location=rtsp://172.17.0.3:8554/mystream'
    # )
    return (
        f'appsrc ! videoconvert' + \
        f' ! queue'
        f' ! x264enc speed-preset=ultrafast tune=zerolatency' + \
        f' ! rtspclientsink protocols=tcp location={stream_uri}'
    )
        

if __name__ == "__main__":
    opt = opts().init()
    main(opt)
