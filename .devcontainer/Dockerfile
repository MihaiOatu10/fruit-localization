FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libsm6 \
    libxext6 \
    glib2.0-0 \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

# Project Python dependencies (build context is the repo root, see devcontainer.json).
# numpy<2 / opencv<4.11: the base image's torch 2.2 is built against NumPy 1.x, and newer OpenCV wheels require NumPy 2.
# opencv-python (also pulled in by ultralytics) shares the cv2 package with opencv-contrib-python, so the contrib
# build is reinstalled last to keep cv2.ximgproc (WLS filter) available.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt "numpy<2" "opencv-python<4.11" "opencv-contrib-python<4.11" \
    && pip install --no-cache-dir --no-deps --force-reinstall "opencv-contrib-python<4.11" \
    && python -c "import cv2, numpy, torch, ultralytics, sklearn, scipy; assert hasattr(cv2, 'ximgproc'); print('cv2', cv2.__version__, 'numpy', numpy.__version__, 'torch', torch.__version__)" \
    && rm /tmp/requirements.txt

WORKDIR /workspace
