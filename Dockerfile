# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# FROM nvcr.io/nvidia/pytorch:21.10-py3
# RUN rm -rf /opt/pytorch  # remove 1.2GB dir

FROM ubuntu:20.04

# ARG USER=standard
# ARG USER_ID=1000 # uid from the previus step
# ARG USER_GROUP=standard
# ARG USER_GROUP_ID=1000 # gid from the previus step
# ARG USER_HOME=/home/${USER}
# # create a user group and a user (this works only for debian based images)
# RUN groupadd --gid $USER_GROUP_ID $USER \
#     && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

ARG DEBIAN_FRONTEND=noninteractive

# Install linux packages
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y python3-pip \
    git \
    zip \
    curl \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libpython3.8-dev \
    python-is-python3

# Install python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext Pillow
RUN pip install --no-cache -r requirements.txt Pillow>=9.1.0 \
    torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install base dependencies + gstreamer
RUN pip uninstall -y opencv-python opencv-python-headless
RUN apt update && apt install -y ffmpeg

RUN \
    apt install -y build-essential \
    cmake \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \ 
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \ 
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev

RUN \
    apt install -y libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-doc \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    gstreamer1.0-rtsp \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    protobuf-compiler \
    libgtk2.0-dev \
    ocl-icd-opencl-dev

# Clone OpenCV repo
WORKDIR /
RUN git clone https://github.com/opencv/opencv.git
WORKDIR /opencv
RUN git checkout 4.5.4

# # Build OpenCV
RUN mkdir /opencv/build 
WORKDIR /opencv/build
RUN ln -s /opt/conda/lib/python3.8/site-packages/numpy/core/include/numpy /usr/include/numpy
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=$(which python) \
    -D BUILD_opencv_python2=OFF \
    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_EXAMPLES=ON ..
RUN make -j$(nproc)

# Install OpenCV
RUN make install
RUN ldconfig

# Create working directory
RUN mkdir -p /app
WORKDIR /app

# set container user
# USER $USER

# Copy contents
COPY . /app

# CMD ["sh", "init_script.sh"]

