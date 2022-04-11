# FROM ubuntu:20.04

# RUN apt-get update 
# RUN apt-get upgrade -y
# # RUN \
# #     DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0

# # Install gstreamer and opencv dependencies
# RUN \
# 	DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# RUN \
# 	apt-get install -y \
# 	libgstreamer1.0-0 \
# 	gstreamer1.0-plugins-base \
# 	gstreamer1.0-plugins-good \
# 	gstreamer1.0-plugins-bad \
# 	gstreamer1.0-plugins-ugly \
# 	gstreamer1.0-libav \
# 	gstreamer1.0-doc \
# 	gstreamer1.0-tools \
#     gstreamer1.0-rtsp \
# 	libgstreamer1.0-dev \
# 	libgstreamer-plugins-base1.0-dev

# RUN apt-get install -y git

# # setup python
# RUN apt-get install -y python3-pip

# # install mavlink dependencies: https://github.com/ArduPilot/pymavlink
# RUN apt-get install -y gcc python3-dev libxml2-dev libxslt-dev

# RUN pip3 install numpy future lxml pymavlink

# # get opencv and build it
# RUN git clone https://github.com/opencv/opencv.git

# RUN apt-get install -y build-essential libssl-dev

# RUN apt-get -y install cmake

# RUN \
# 	cd opencv && \
# 	git checkout 4.5.4 && \
# 	git submodule update --recursive --init && \
# 	mkdir build && \
# 	cd build && \
# 	cmake -D CMAKE_BUILD_TYPE=RELEASE \
# 	-D INSTALL_PYTHON_EXAMPLES=ON \
# 	-D INSTALL_C_EXAMPLES=OFF \
# 	-D PYTHON_EXECUTABLE=$(which python3) \
# 	-D BUILD_opencv_python2=OFF \
# 	-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
# 	-D PYTHON3_EXECUTABLE=$(which python3) \
# 	-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
# 	-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
# 	-D WITH_GSTREAMER=ON \
# 	-D BUILD_EXAMPLES=ON .. && \
# 	make -j$(nproc) && \
# 	make install && \
# 	ldconfig

# # Install python dependencies
# COPY requirements.txt .
# RUN python3 -m pip install --upgrade pip
# RUN pip3 install --no-cache torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# RUN pip3 install --no-cache -r requirements.txt

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

ARG USER=standard
ARG USER_ID=1000 # uid from the previus step
ARG USER_GROUP=standard
ARG USER_GROUP_ID=1000 # gid from the previus step
ARG USER_HOME=/home/${USER}
# create a user group and a user (this works only for debian based images)
RUN groupadd --gid $USER_GROUP_ID $USER \
    && useradd --uid $USER_ID --gid $USER_GROUP_ID -m $USER

# Install linux packages
RUN apt-get update && apt-get upgrade -y
RUN \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext
RUN pip install --no-cache -r requirements.txt \
    torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install base dependencies + gstreamer
# RUN pip uninstall -y opencv-python
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install ffmpeg

RUN \
    DEBIAN_FRONTEND=noninteractive \
    apt-get -y install build-essential \
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
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y libgstreamer1.0-0 \
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
    cmake \
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
USER $USER

# Copy contents
# COPY . /app

# CMD ["sh", "init_script.sh"]

