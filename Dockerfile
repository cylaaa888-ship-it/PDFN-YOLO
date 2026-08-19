FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/pdfn
COPY . /workspace/pdfn
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install -e .
CMD ["python3", "scripts/model_info.py"]
