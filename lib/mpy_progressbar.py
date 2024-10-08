
# 思路来自：https://stackoverflow.com/questions/69423410/moviepy-getting-progress-bar-values

from proglog import ProgressBarLogger

class mpy_qt_progressbar(ProgressBarLogger):

    def __init__(self,qt_bar_callback:any):
        super().__init__()
        self.qt_bar_callback = qt_bar_callback
        self.last_progress_int = 0
        

    def callback(self, **changes):
        # Every time the logger message is updated, this function is called with
        # the `changes` dictionary of the form `parameter: new value`.

        # for (parameter, value) in changes.items():
        #     print ('Parameter %s is now %s' % (parameter, value))
        pass

    def set_qt_bar_callback(self,qt_bar_callback:any):
        self.qt_bar_callback = qt_bar_callback

    def bars_callback(self, bar, attr, value,old_value=None):
        # Every time the logger progress is updated, this function is called        
        progress_int = int((value / self.bars[bar]['total']) * 100)
        if progress_int != self.last_progress_int:
            self.last_progress_int = progress_int
            self.qt_bar_callback(progress_int)
