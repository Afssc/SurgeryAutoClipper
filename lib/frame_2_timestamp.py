def frame2timestamp(frame_range, fps):
    """
    
    将frame转换为时间戳
    
    args:
        frame_range: list of frames (start, end)
        fps: int, fps of the video
    
    return:
        list of timestamps [(h,m,s),(h,m,s)]
    
    """
    start_time = 1000/fps * frame_range[0] 
    end_time = 1000/fps * frame_range[1] 
    m1, s1 = divmod(start_time // 1000, 60)
    h1, m1 = divmod(m1, 60)
    m0, s0 = divmod(end_time // 1000, 60)
    h0, m0 = divmod(m0, 60)
    return [(h1,m1,s1),(h0,m0,s0)]