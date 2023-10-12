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
        dataset = LoadStreams(opt.source, opt.vstream_uri)
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
                anonybboxes = []

                if webcam:  # batch_size >= 1
                    p, im0 = path[i], im0s[i].copy()
                else:
                    p, im0 = path, im0s.copy()

                pbar.set_description("Anonymizing %s" % p)

                # Inference
                p = Path(p)  # to Path
                save_path = str(opt.save_dir / p.name)  # im.jpg

                imc = im0.copy()
                # Head detection
                head_pred = head_detector.process_im(imc, head_classes)
                # Process predictions
                det = head_pred[0]  # per image
                if len(det):
                    for *head_xyxy, head_conf, head_cls in reversed(det):
                        head_bbox = (
                            torch.tensor(head_xyxy)
                            .view(1, 4)
                            .clone()
                            .view(-1)
                            .numpy()
                            .astype(np.int32)
                        )
                        # miny, maxy, minx, maxx
                        anonybbox = (
                            head_bbox[1],
                            head_bbox[3],
                            head_bbox[0],
                            head_bbox[2],
                        )
                        anonybboxes.append(anonybbox)

                imc = im0.copy()
                # Vehicle detection
                if opt.two_stage:
                    veh_pred = veh_detector.process_im(imc, veh_classes)
                    det = veh_pred[0]  # per image
                    if len(det):
                        for *veh_xyxy, veh_conf, veh_cls in reversed(det):
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
                            lp_pred = lp_detector.process_im(veh_imc, lp_classes)
                            det = lp_pred[0]  # per image
                            if len(det):
                                for *lp_xyxy, lp_conf, lp_cls in reversed(det):
                                    lp_bbox = (
                                        (torch.tensor(lp_xyxy).view(1, 4))
                                        .view(-1)
                                        .numpy()
                                        .astype(np.int32)
                                    )
                                    # miny, maxy, minx, maxx
                                    anonybbox = (
                                        veh_bbox[1] + lp_bbox[1],
                                        veh_bbox[1] + lp_bbox[3],
                                        veh_bbox[0] + lp_bbox[0],
                                        veh_bbox[0] + lp_bbox[2],
                                    )
                                    anonybboxes.append(anonybbox)

                else:
                    imc = im0.copy()
                    # License plate detection
                    lp_pred = lp_detector.process_im(imc, lp_classes)
                    det = lp_pred[0]  # per image
                    if len(det):
                        for *lp_xyxy, lp_conf, lp_cls in reversed(det):
                            lp_bbox = (
                                (torch.tensor(lp_xyxy).view(1, 4))
                                .view(-1)
                                .numpy()
                                .astype(np.int32)
                            )
                            # miny, maxy, minx, maxx
                            anonybbox = (lp_bbox[1], lp_bbox[3], lp_bbox[0], lp_bbox[2])
                            anonybboxes.append(anonybbox)

                imc = im0.copy()
                w, h = im0.shape[1], im0.shape[0]
                for miny, maxy, minx, maxx in anonybboxes:
                    miny = max(0, min(miny, h - 1))
                    maxy = max(0, min(maxy, h - 1))
                    minx = max(0, min(minx, w - 1))
                    maxx = max(0, min(maxx, w - 1))
                    if maxy > miny and maxx > minx:
                        sub_im = imc[miny : (maxy + 1), minx : (maxx + 1)]
                        sub_im = cv2.GaussianBlur(sub_im, (45, 45), 30)
                        im0[
                            miny : (maxy + 1),
                            minx : (maxx + 1),
                        ] = sub_im
                
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
                            fps = (
                                opt.stream_fps
                                if opt.stream_fps is not None
                                else dataset.fps[i]
                            )
                            w, h = im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        if opt.vstream_uri is not None:
                            vid_writer[i] = cv2.VideoWriter(
                                gstreamer_pipeline_out(dataset.streams[i]),
                                cv2.CAP_GSTREAMER,
                                0,
                                fps,
                                (w, h),
                                True,
                            )
                        else:
                            vid_writer[i] = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                            )
                        
                    if not vid_writer[i].isOpened():
                        raise Exception("can't open video writer")
                    vid_writer[i].write(im0)
                    # print("frame written to the server")

                pbar.update(1)


def gstreamer_pipeline_out(vstream_uri):
    # return (
    #     f'appsrc ! videoconvert' + \
    #     f' ! x264enc speed-preset=ultrafast tune=zerolatency' + \
    #     f' ! rtspclientsink protocols=tcp location={stream_uri}'
    # )
    # return (
    #     'appsrc name=appsrc format=time is-live=true caps=video/x-raw,format=(string)BGR appsrc. ! videoconvert' + \
    #     ' ! x264enc tune=zerolatency' + \
    #     ' ! rtspclientsink protocols=tcp location=rtsp://172.17.0.3:8554/mystream'
    # )
    # if astream_uri is not None:
    #     return (
    #         f"appsrc"
    #         + f" ! videoconvert"
    #         + f" ! x264enc speed-preset=ultrafast tune=zerolatency"
    #         + f" ! rtspclientsink protocols=tcp location={vstream_uri}"
    #         + f" rtspsrc location={source}"
    #         + f" ! rtpmp4gdepay ! aacparse"
    #         + f" ! rtspclientsink protocols=tcp location={astream_uri}"
    #     )
    # else:
    #     return (
    #         f"appsrc"
    #         + f" ! videoconvert"
    #         + f" ! x264enc speed-preset=ultrafast tune=zerolatency"
    #         + f" ! rtspclientsink protocols=tcp location={vstream_uri}"
    #     )
    return (
        f"appsrc"
        + f" ! videoconvert"
        + f" ! x264enc speed-preset=ultrafast tune=zerolatency"
        + f" ! rtspclientsink protocols=tcp location={vstream_uri}"
    )
    # gst-launch-1.0 rtspsrc location=rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k ! rtph264depay ! h264parse ! rtspclientsink protocols=tcp location=rtsp://172.17.0.3:8554/VideoStream rtspsrc location=rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k ! rtpmp4gdepay ! aacparse ! rtspclientsink protocols=tcp location=rtsp://172.17.0.3:8554/AudioStream


if __name__ == "__main__":
    opt = opts().init()
    main(opt)
