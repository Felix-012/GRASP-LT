# docker build -f environment/Dockerfile -t grasp .
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip git \
      libglib2.0-0 libgl1 libhdf5-dev libmetis-dev && rm -rf /var/lib/apt/lists/*
RUN python3.11 -m venv /opt/grasp && /opt/grasp/bin/pip install --upgrade pip
ENV PATH=/opt/grasp/bin:$PATH
COPY environment/requirements.txt /tmp/requirements.txt
RUN pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128 \
 && pip install -r /tmp/requirements.txt && (pip install metis==0.2a5 || true)
WORKDIR /workspace/GRASP
