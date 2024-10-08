# -*- encoding: utf-8 -*-
'''
@File    :   concatenator.py
@Time    :   2024/09/23 14:05:34
@Author  :   AFSSC 
@Version :   1.0
@Contact :   zyj@shu.edu.cn
'''

# here put the import lib

# import cv2
# import moviepy.editor as mpy
from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from .frame_2_timestamp import frame2timestamp

class concatenator:
    def __init__(self, filename,output_path,logger):
        self.logger = logger
        self.filename = filename
        self.orginal_video = VideoFileClip(self.filename)
        self.output_path = output_path
        self.output_size = self.orginal_video.size
        self.fps = self.orginal_video.fps

    def to_video_fileclips(self, frame_list):
        """

        转换到video file clips

        args:
            frame_list: list of frames [(start, end), (start, end), ...]
        
        """
        clips = []
        for frame_range in frame_list:
            start, end = frame2timestamp(frame_range, self.fps)
            clip = self.orginal_video.subclip(start, end)
            # clip = clip.resize(self.output_size)
            clip = resize(clip, self.output_size)
            clips.append(clip)
        return clips

    def concatenate(self, clips) -> None:
        """

        拼接视频

        args:
            clips: list of clips
        
        """
        mpyclips = self.to_video_fileclips(clips)
        final_clip = concatenate_videoclips(mpyclips)
        final_clip.write_videofile(self.output_path, fps=self.fps, codec='libx264',logger=self.logger)
        
    def __del__(self):
        self.orginal_video.close()