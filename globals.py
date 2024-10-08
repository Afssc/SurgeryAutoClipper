from typing import TypedDict,List
from cv2 import VideoCapture
class KeyFrameDict(TypedDict):
    clip:int
    keyframe:List[int]

# 全局变量们
cap_g:VideoCapture = None
filepath_g:str = ""
crude_range_g = []
key_frame_g = KeyFrameDict()
final_range_g = []
