# -*- encoding: utf-8 -*-
'''
@File    :   crude_cutter.py
@Time    :   2024/09/12 01:15:53
@Author  :   AFSSC 
@Version :   1.0
@Contact :   zyj@shu.edu.cn
'''


import cv2
import numpy as np
import logging as log
import typing as t
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from .abstract_cutter import cutter
import json

log.debug = print
log.critical = print
log.warning = print

class CrudeCutter(cutter):

    """
    粗剪剪辑器

    args:
        video_path(str) : 视频路径
        output_path(str) : 输出路径

    """

    def __init__(self, video_path: str,logger = None,progressbar_callback = None) -> None:

        self.log = logger if logger else print
        self.progressbar_callback = progressbar_callback if progressbar_callback else lambda x:x
        self.video_path = video_path
        self.template_path = R"templates\endoscope.png"
        self.template_threshold = 50000
        self.proxy_size = (214, 120)

        # 打开视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.log("Cannot open video.")
        self.log(f"视频文件:{video_path} 已打开")

        self.total_frame = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化模板
        self.tplt = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        self.tplt[np.where(self.tplt == 0 )] = 1
        self.tplt[np.where(self.tplt == 50 )] = 0
        self.tplt[np.where(self.tplt == 150 )] = 0
        self.tplt[np.where(self.tplt == 250 )] = 0


        # 得分列表
        self.match_score = [] 
        self.match_frame = []
        self.match_edges = []

        self.accurate_edges = []

        #设置检测参数
        self.tgt_frame_num = 50
        self.gaussianBlur_kernel_size = [5,5]
        self.dilate_iteration = 2



    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()
        del self.match_frame
        del self.match_score
        del self.match_edges
        del self.accurate_edges
        self.log("资源已释放。")


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


    def _calculate_match_score(self,src: np.ndarray,template: np.ndarray) -> int:

        """
        计算匹配得分。说是卷积其实就是整个图片点乘。
        args:
            src(np.ndarray) : 原始帧
            template(np.ndarray) : 模板
        """
        return (template*src).sum() 
    

    def _crude_search_preprocess(self,frame: np.ndarray) -> np.ndarray:

        """
        预处理管线

        args:
            frame(np.ndarray) :原始帧
        """
        frame = cv2.resize(frame, self.proxy_size, None)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame,tuple(self.gaussianBlur_kernel_size),0)
        frame = cv2.dilate(frame, None, iterations=self.dilate_iteration)
        frame = cv2.threshold(frame, 30, None, cv2.THRESH_TOZERO, None)[1]
        frame = frame // 10
        return frame
    

    def _crude_search(self) -> None:

        """
        粗剪采样。

        采样数量 = tgt_frame_num
        """

        step = int(self.total_frame//self.tgt_frame_num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last_judge = False
        last_frame = 0
        self.match_score = []
        self.match_frame = []
        self.match_edges = []

        for i in tqdm.tqdm(range(0,int(self.total_frame),step)):
            self.progressbar_callback(5+int(i/self.total_frame*75))
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self._crude_search_preprocess(frame)
            score = self._calculate_match_score(frame,self.tplt)
            self.match_score.append(score)
            self.match_frame.append(i - step)
            if last_judge ^ (score < self.template_threshold):
                self.match_edges.append((last_frame - step,i - step,True if last_judge else False))
            last_judge = score < self.template_threshold
            last_frame = i
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)


    def _binary_search(self,start_index: int, end_index: int , is_upside:bool) -> int:
        
        """
        二分法搜索。

        args:

            start_index(int) : 起始帧引索
            end_index(int) : 结束帧引索
            is_upside(bool) : 上升沿/下降沿
            cap(cv2.VideoCapture) : 视频捕获器
            template(np.ndarray) : 匹配模板
            template_threshold(int) : 匹配模板阈值
        
        return:

            int : 关键帧引索
        """

        l = start_index 
        h = end_index
        # print(f"h:{h},l:{l}")
        while(h - l > 1):
            # print(f"h:{h},l:{l}")
            m = (h+l)//2
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, m)
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self._crude_search_preprocess(frame)
            # show_frame(frame)
            score = self._calculate_match_score(frame,self.tplt)
            # self.log(f"middle:%7d,  score:%7d,  {score > self.template_threshold}"%(m,score))
            if is_upside:
                if score > self.template_threshold:
                    h = m
                else:
                    l = m
            else:
                if score > self.template_threshold:
                    l = m
                else:
                    h = m
        return h
    
    
    def _find_accurate_edges(self) -> None:

        """
        二分法查找精确边界。
        """

        self.accurate_edges = []
        cnt = 0
        for edge in self.match_edges:
            self.progressbar_callback(80+int(cnt/(len(self.match_edges))*15))
            self.accurate_edges.append((self._binary_search(*edge),edge[2]))
            self.log(f"已找到片段边界{cnt}")
            cnt += 1


    def clip(self) -> tuple:
        """
        求解粗剪剪辑区间。

        return:

            wanted_clips(tuple) : ((start_frame,end_frame),...)
            timeline(tuple) : ((start_time,end_time),...)
        """
        self.log(f"采样帧数 = {self.tgt_frame_num}。\n采样中...")
        self._crude_search()
        if not self.match_edges:
            self.log("未发现可选取片段。")
            return None,None
        self.log(f"采样完成。\n发现{len(self.match_edges)}次跳变。\n查找精确边界中...")
        self._find_accurate_edges()
        self.log(f"查找完成。")
        wanted_clips = []
        timeline = []

        clip_start = -1
        for edge in self.accurate_edges:
            if edge[1] == 0:
                clip_start = edge[0]
            else:
                if clip_start != -1:
                    wanted_clips.append((clip_start,edge[0]))
                    clip_start = -1
                else:
                    self.log(f"Double upside edge at {edge[0]}")
        
        if clip_start != -1:
            wanted_clips.append((clip_start,int(self.total_frame)))

        for clip in wanted_clips:
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, clip[0])
            # clip_start_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            # self.cap.set(cv2.CAP_PROP_POS_FRAMES, clip[1])
            # clip_end_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            # 你们难道都是用fps算的时间戳？为什么cap.get返回的时间不对？

            clip_start_time = 1000/self.fps * clip[0] 
            clip_end_time = 1000/self.fps * clip[1] 
            timeline.append((clip_start_time,clip_end_time))
        self.log("区间提取完成。")
        self.progressbar_callback(99)
        return wanted_clips,timeline
    

    

if __name__ == "__main__":
    cutter = CrudeCutter(R"D:\AFSSC\Documents\python学习\视频剪辑项目\手术原始视频.mp4")
    clips = cutter.clip()
    print(clips)
    for timel in clips[1]:
        m1, s1 = divmod(timel[0] // 1000, 60)
        h1, m1 = divmod(m1, 60)
        m0, s0 = divmod(timel[1] // 1000, 60)
        h0, m0 = divmod(m0, 60)
        print ("%02d:%02d:%02d  ~  %02d:%02d:%02d" % (h1, m1, s1, h0, m0, s0))
    del cutter

        

