#!/bin/bash

# 初始化 conda
source /opt/conda/etc/profile.d/conda.sh

# 激活指定的 conda 环境（假设环境名为 model_env）
conda activate taming_3dgs

# 设置环境变量
export PYTHONPATH=/app
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

# 检查是否提供了必需的参数
if [ "$1" = "-zip-url" ] && [ -n "$2" ] && [ "$3" = "-job-id" ] && [ -n "$4" ]; then
    ZIP_URL=$2
    JOB_ID=$4
else
    echo "Usage: $0 -zip-url <zip_url> -job-id <job_id>"
    exit 1
fi

# 创建临时目录来存储下载的文件
TEMP_DIR="/app/temp_data"
mkdir -p $TEMP_DIR

# 下载 zip 文件
DOWNLOADED_ZIP="$TEMP_DIR/source.zip"
wget -O "$DOWNLOADED_ZIP" "$ZIP_URL"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download zip from $ZIP_URL"
    exit 1
fi

# 解压 zip 文件
unzip -o "$DOWNLOADED_ZIP" -d "$TEMP_DIR"
if [ $? -ne 0 ]; then
    echo "Error: Failed to unzip file"
    exit 1
fi

# 设置虚拟显示器
Xvfb :0 -screen 0 1024x768x24 &

# 打印下解压后的目录
ls -R "$TEMP_DIR"

# 使用解压后的目录路径运行原有脚本
python convert_custom.py -s "$TEMP_DIR"
python train_custom.py -s "$TEMP_DIR" --job_id $JOB_ID 

# 设置 AWS 默认区域
export AWS_DEFAULT_REGION=us-east-1

# 获取当前日期，格式为 yyyymmdd
DATE=$(date +%Y%m%d)

cd ./output/$JOB_ID/point_cloud/iteration_30000
# 上传到指定的 S3 存储桶，包含日期路径
aws s3 cp ./point_cloud.ply s3://future-gadgets-resource-01/point_cloud/$DATE/$JOB_ID/iteration_30000/point_cloud.ply

# 可选：添加上传成功确认信息
if [ $? -eq 0 ]; then
    echo "Successfully uploaded point cloud to S3 bucket"
else
    echo "Failed to upload point cloud to S3 bucket"
    exit 1
fi