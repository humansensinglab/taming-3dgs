# 使用 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

# 避免交互式配置
ENV DEBIAN_FRONTEND=noninteractive

# 设置 CUDA 架构
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    python3-pip \
    git \
    wget \
    cmake \
    build-essential \
    ninja-build \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.7 为默认版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# 安装 pip for Python 3.7
RUN wget https://bootstrap.pypa.io/pip/3.7/get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py

# 创建并激活 Conda 环境
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# 设置工作目录
WORKDIR /app

# 复制所有源代码文件
COPY . /app/

# 创建基础 conda 环境并安装 PyTorch
RUN conda create -n taming_3dgs python=3.7.13 && \
    conda install -n taming_3dgs pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=11.6 ninja -c pytorch -c conda-forge

# 激活环境并安装其他依赖
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate taming_3dgs && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=$CUDA_HOME/bin:$PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6" && \
    conda env update -f environment.yml

# 创建用于挂载外部文件的目录
RUN mkdir -p /app/input /app/output

# 在 Dockerfile 中添加环境变量设置
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV PYTHONPATH=/app:$PYTHONPATH

# 安装 Flask
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate taming_3dgs && \
    pip install flask

# 修改启动脚本
RUN echo '#!/bin/bash' > /app/run.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /app/run.sh && \
    echo 'conda activate taming_3dgs' >> /app/run.sh && \
    echo 'export CUDA_VISIBLE_DEVICES=0' >> /app/run.sh && \
    echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> /app/run.sh && \
    echo 'export PYTHONPATH=/app:$PYTHONPATH' >> /app/run.sh && \
    echo 'python server.py' >> /app/run.sh && \
    chmod +x /app/run.sh

# 暴露端口
EXPOSE 5000

# 设置容器入口点
ENTRYPOINT ["/app/run.sh"]

