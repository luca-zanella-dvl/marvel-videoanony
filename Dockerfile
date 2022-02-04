# Start FROM Python official image
FROM python:3.8-slim-buster

# Install linux packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev libglib2.0-0

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install --no-cache -r requirements.txt

# Create working directory
RUN mkdir -p /app
WORKDIR /app

# Copy contents
COPY . /app

CMD ["sh", "init_script.sh"]