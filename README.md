# SurgeryAutoClipper  ![Static Badge](https://img.shields.io/badge/python-3.10-blue)  [![License: MIT](https://img.shields.io/badge/license-MIT-red.svg)](https://opensource.org/licenses/MIT)
基于opencv的手术自动剪辑工具，带有qt图形界面
  
## 功能
#### 1. 粗剪
  * 自动采样整个视频，根据模板匹配打分
  * 在分数边沿二分搜索精确区间边界，得到有效区间
#### 2. 精剪
  * 用户在有效区间内手动选择关键帧
  * resnet模型去掉FC作为特征提取器
  * 在关键帧附近拾取特征，计算余弦相似度
  * 阈值选出最终片段
#### 3.导出
  * 用户勾选最终片段，拼接导出到用户指定目录

## 安装
推荐使用虚拟环境
```shell
git clone https://github.com/Afssc/SurgeryAutoClipper.git
cd SurgeryAutoClipper
pip install -r requirement.txt
```

## 运行
```shell
python gui.py
```

## 目录结构
```
    Surgery-AutoClip
    |
    │  config.json              配置文件
    │  globals.py               全局变量
    │  gui.py                   主界面（从此启动）
    │  MainWindow.ui            qml
    │  Ui_MainWindow.py         主窗口基类
    │  requirements.txt         第三方库依赖
    │
    ├─lib
    │  │  abstract_cutter.py    剪辑器基类
    │  │  concatenator.py       合并片段工具
    │  │  crude_cutter.py       粗剪剪辑器
    │  │  fine_cutter.py        精剪剪辑器
    │  │  frame_2_timestamp.py  帧位置转时间戳
    │  │  mpy_progressbar.py    qt进度条支持
    │  └─ __init__.py
    │
    ├─templates                 
    │      endoscope.png        模板样式（内窥镜）
    │
    ├─testify                   可行性验证
    │      demo.ipynb
    │      demo.log
    │      resnet.ipynb
    │      widgettest.py
    │
    └─tmp                       拼接临时文件目录
```

## 评价
#### 优势：
  * 灵活性高，易扩展，不需要预训练神经网络分类器，增加样式需求时只需添加png格式模板
  * 速度快，无需遍历视频文件，对于长片段手术视频只需采样50帧就能选出有效片段
  * 给予用户选择关键帧和最终的自由
#### 劣势：
  * 目前仅支持resnet和余弦相似度，HOG-SVM等其他算法暂未实现