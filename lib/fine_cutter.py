# -*- encoding: utf-8 -*-
'''
@File    :   fine_cutter.py
@Time    :   2024/09/18 10:21:09
@Author  :   AFSSC 
@Version :   1.0
@Contact :   zyj@shu.edu.cn
'''

# here put the import lib

import cv2
import tqdm
import numpy as np
from abc import ABC, abstractmethod
import logging as log
import matplotlib.pyplot as plt
from .abstract_cutter import cutter
import json

log.debug = print

class algorithms(enumerate):
    RESNET = 0
    ORB = 1
    SIFT = 2
    SURF = 3


class FineCutter(cutter):

    """

    精剪剪辑器
    """
    

    def __init__(self, video_path: str,algo:algorithms,
                 start:int,end:int,std_index: int,sample_num:int,
                 logger = None, processbar_callback = None) -> None:
        self.log = logger if logger else print
        self.processbar_callback = processbar_callback if processbar_callback else lambda x:x
        self.processbar_callback(0)
        self.video_path = video_path
        self.template_path = R"templates\endoscope.png"     #默认线程从主函数位置启动
        self.proxy_size = (214, 120)

        # 打开视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.log("无法打开视频。可能文件已经被移动或删除。")
            raise FileNotFoundError

        self.total_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vstart = start
        self.vend = end
        self._algo = algo
        self._interval = 50
        self._std = std_index
        self._samplt_num = sample_num
        self._reserve = 50
        self._offset = int(sample_num * self._interval//2)

        # 初始化模板（彩）
        tplt = cv2.imread(self.template_path, cv2.IMREAD_COLOR)
        tplt[np.where(tplt == 0 )] = 0
        tplt[np.where(tplt == 50 )] = 0
        tplt[np.where(tplt == 150 )] = 255
        tplt[np.where(tplt == 250 )] = 0
        tplt = cv2.resize(tplt,(self.width,self.height))
        tplt[np.where(tplt != 255 )] = 0
        self.tplt = tplt

        self.match_score = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._std)
        __std = self.cap.read()[1]
        __std = self._masking(__std,self.tplt)
        __std = cv2.cvtColor(__std, cv2.COLOR_BGR2RGB)

        if (algo == algorithms.RESNET):
            self.log("使用特征提取器: resnet\n加载模型中...")
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms
            from PIL import Image

            self.nograde = torch.no_grad
            self.pilimg = Image
            self.calculate_with_current_strategy = self._strategy_RESNET
            # self.device = torch.device("cpu")
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])  # 去掉最后的全连接层
            self.model.eval()
            # 预处理
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.cos = nn.CosineSimilarity(dim=0)
            self.threshold = 0.8
            self.std_feature = self._extract_features(__std)
            # self.log(f"决断阈值 = {self.threshold}")

        else:
            self.log("未实现的算法")
            raise NotImplementedError
        
        
    @property
    def algo(self):
        return self._algo
    
    @property
    def std_index(self):
        return self._std
    
    @algo.setter
    def algo(self,algo:algorithms):
        self._algo = algo
    
    @std_index.setter
    def std_index(self,std_index:int):
        self._std = std_index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._std)
        __std = self.cap.read()
        __std = self._masking(__std[1],self.tplt)
        if (self._algo == algorithms.RESNET):
            self.std_feature = self._extract_features(__std)

    def from_json(self,json_filepath:str) -> None:
        try:
            with open(json_filepath,"r") as f:
                json_dict = json.load(f)["crude_cutter_cfg"]
                self.from_dict(json_dict)
                self.log("配置文件已加载")
        except Exception as e:
            self.log(f"读取配置文件失败：{e}")

    def show_frame(self,f: np.ndarray,delay:int = 0) -> None:
        # resized.get_frame(0).shape
        cv2.imshow("frame", f )
        cv2.waitKey(delay)
        if delay == 0:
            cv2.destroyAllWindows()

    def _extract_features(self,img: np.ndarray):
        img = self.pilimg.fromarray(img)
        img = self.preprocess(img).unsqueeze(0)
        with self.nograde():
            features = self.model(img)
        return features.squeeze()
    
    def _cosine_similarity(self,feature1, feature2) -> float:
        
        return self.cos(feature1, feature2).item()
    
    def _cosine_similarity_with_std(self, feature) -> float:

        return self.cos(feature, self.std_feature).item()
    
    def _masking(self,f,msk) -> np.ndarray:
        # f = cv2.resize(f, (tplt.shape[1], tplt.shape[0]))
        # msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
        # self.show_frame(msk,0)
        # self.show_frame(f,0)
        f = cv2.bitwise_and(f, msk)
        return f

    def _strategy_RESNET(self) -> None:

        self.match_score = []
        for i in tqdm.tqdm(range(self._std - self._offset,self._std + self._offset,self._interval)):
            self.processbar_callback(int((i - self._std + self._offset)/(2*self._offset)*100))
            if i < self.vstart or i > self.vend:
                self.match_score.append(0)
                continue
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            frame = self.cap.read()[1]
            if frame is None:
                break
            frame = self._masking(frame,self.tplt)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features = self._extract_features(frame)
            similarity = self._cosine_similarity_with_std(features)
            self.match_score.append(similarity)


    def _strategy_ORB(self) -> None:
        raise NotImplementedError
    
    def _strategy_SIFT(self) -> None:
        raise NotImplementedError
    
    def _strategy_SURF(self) -> None:
        raise NotImplementedError
    
    def clip(self) -> tuple:
        self.calculate_with_current_strategy()
        # plt.plot(self.match_score)
        # plt.show()
        left_cliff = 0
        right_cliff = len(self.match_score)

        sorted_score = sorted(self.match_score)
        score_threshold = sorted_score[int(len(sorted_score) * (1 - self.threshold))]

        for i in range(len(self.match_score)//2 - self._reserve //2 ,0,-1):
            if self.match_score[i] < score_threshold:
                left_cliff = i
                break

        for i in range(len(self.match_score)//2 + self._reserve //2 ,len(self.match_score),1):
            if self.match_score[i] < score_threshold:
                right_cliff = i
                break
        
        left_index = left_cliff * self._interval + self._std - self._offset
        right_index = right_cliff * self._interval + self._std - self._offset

        clip_start_time = 1000/self.fps * left_index
        clip_end_time = 1000/self.fps * right_index

        return (left_index,right_index) , (clip_start_time,clip_end_time)

    def __del__(self):
        self.cap.release()

    # def __call__(self):
    #     self.clip()

if __name__ == "__main__":
    fc = FineCutter(R"D:\AFSSC\Documents\python学习\视频剪辑项目\手术原始视频.mp4",
                    algorithms.RESNET,100000,191981,114514,200)
    print(fc.clip())

    del fc
