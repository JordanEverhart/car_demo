#!/bin/bash
# 进入 ROS 环境
source /opt/ros/noetic/setup.bash

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# 运行键盘控制脚本
python3 keyboard_control.py

