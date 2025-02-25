#!/bin/bash

SOURCE_PATH=$1

Xvfb :0 -screen 0 1024x768x24 &
export DISPLAY=:0
python convert_custom.py -s $SOURCE_PATH

python train_custom.py -s $SOURCE_PATH