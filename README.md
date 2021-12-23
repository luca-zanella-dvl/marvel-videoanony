<div align="center">

<p>
   <img src="https://github.com/luca-zanella-dvl/marvel-video-anonymization/blob/master/images/anonymized.gif" width="850" />
</p>
   
<br>
<p>
Our repo is used for head and license plate anonymization. We first use a fine-tuned yolov5 for head/license plate detection and then apply blurring to obfuscate the heads/license plates.
</p>

</div>

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/luca-zanella-dvl/marvel-video-anonymization/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/luca-zanella-dvl/marvel-video-anonymization
$ cd marvel-video-anonymization
$ conda create --name marvel-video-anonymization python=3.9
$ conda activate marvel-video-anonymization
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Pretrained Models</summary>
   
First, create the `weights` folder in the root project directory.
   
```bash
$ mkdir weights
```
Then, please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1YfZ-WSh5W1fCnje4fMgaY9EsXH2xMNnP?usp=sharing) and
save them under the `weights` folder.

</details>

<details open>
<summary>Anonymize with anonymize.py</summary>
   
`anonymize.py` runs heads and license plates anonymization on a variety of sources, saving results to `runs/anonymize`.
  
Run commands below to anonymize heads.
   
```bash
$ python anonymize.py --weights weights/crowdhuman1280x_yolov5x6.pt --classes 0 --imgsz 1280 --anonymize-heads --source 0  # webcam 
                                                                                                                        img.jpg  # image
                                                                                                                        vid.mp4  # video
                                                                                                                        path/  # directory
                                                                                                                        path/*.jpg  # glob
                                                                                                                        'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                                                                        'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

Run commands below to anonymize license plates.
   
```bash
$ python anonymize.py --weights weights/yolov5x6.pt --lpd-weights lpd_yolov5x.pt --classes 2 3 7 --imgsz 1280 --lpd-imgsz 640 --anonymize-lps --source 0  # webcam 
                                                                                                                                                       img.jpg  # image
                                                                                                                                                       vid.mp4  # video
                                                                                                                                                       path/  # directory
                                                                                                                                                       path/*.jpg  # glob
                                                                                                                                                       'https://youtu.be/Zgi9g1ksQHc'  # YouTube                      
                                                                                                                                                       'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```
   
</details>

</div>
