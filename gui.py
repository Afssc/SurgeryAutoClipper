# -*- encoding: utf-8 -*-
'''
@File    :   gui.py
@Time    :   2024/09/23 14:05:43
@Author  :   AFSSC 
@Version :   1.0
@Contact :   zyj@shu.edu.cn
'''

# here put the import lib

import sys
import os
import time
import json
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtWidgets import QMainWindow,QApplication,QWidget,QFileDialog,QTextEdit,QTableWidgetItem
from PyQt5.QtGui import QPixmap,QImage
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QThread,pyqtSignal
from Ui_MainWindow import Ui_MainWindow
import cv2
from abc import ABCMeta,abstractmethod
from typing import List,TypedDict
from lib import mpy_qt_progressbar,concatenator

class KeyFrameDict(TypedDict):
    clip:int
    keyframe:List[int]

# 全局变量们
cap_g:cv2.VideoCapture = None
filepath_g:str = ""
crude_range_g = []
key_frame_g = KeyFrameDict()
final_range_g = []


class cutter_thread(QThread):
    
    update_textedit = pyqtSignal(str)
    update_progressbar = pyqtSignal(int)

    @abstractmethod
    def run(self):
        raise NotImplementedError


class crude_cutter_thread(cutter_thread):

    def run(self):
        logger = self.update_textedit.emit
        progressbar_callback = self.update_progressbar.emit
        time.sleep(.2)
        progressbar_callback(2)
        logger("加载粗剪辑器...")
        from lib import CrudeCutter
        global filepath_g
        crude_cutter = CrudeCutter(filepath_g,logger,progressbar_callback)
        crude_cutter.from_json("config.json")
        time.sleep(.2)
        logger("加载完成。开始处理...")
        time.sleep(.2)
        logger("\n*********************************粗剪线程*********************************\n")
        frame_ranges,timelines = crude_cutter.clip()
        time.sleep(.2)
        logger("\n***********************************结果************************************\n")
        if frame_ranges is None:
            logger("处理失败。")
            self.update_progressbar.emit(100)
            return
        for fr,timel in zip(frame_ranges,timelines):
            m1, s1 = divmod(timel[0] // 1000, 60)
            h1, m1 = divmod(m1, 60)
            m0, s0 = divmod(timel[1] // 1000, 60)
            h0, m0 = divmod(m0, 60)
            logger(f"帧区间：{fr}, 时间轴：%02d:%02d:%02d  ~  %02d:%02d:%02d" % (h1, m1, s1, h0, m0, s0))
        logger("处理完成！")
        global crude_range_g
        crude_range_g = frame_ranges
        self.update_progressbar.emit(100)
                
class refine_cutter_thread(cutter_thread):
    
    update_taskprogress_label = pyqtSignal(str)

    def run(self):

        global final_range_g 
        global filepath_g
        logger = self.update_textedit.emit
        progressbar_callback = self.update_progressbar.emit
        time.sleep(.2)
        task_number = 0
        for i in key_frame_g:
            task_number += len(key_frame_g[i])
        current_task = 0
        self.update_taskprogress_label.emit(f"任务进度：{current_task}/{task_number}")
        logger("加载精剪器...")
        from lib import algorithms,FineCutter
        time.sleep(.2)
        logger("加载完成。开始处理...")
        time.sleep(.2)
        timelines = []
        logger(f"使用视频:{filepath_g}")
        e = None
        logger("\n*********************************精剪线程*********************************\n")
        try:
            for clip,i in zip(crude_range_g,range(len(key_frame_g))):
                for keyframe in key_frame_g[i]:
                    logger(f"处理片段{i+1}，关键帧{keyframe}...")
                    fine_cutter = FineCutter(filepath_g,algorithms.RESNET,*clip,
                                            keyframe,50,logger,progressbar_callback)
                    fine_cutter.from_json("./config.json")
                    logger("精剪剪辑器设置完成，开始处理...")
                    section, timeline = fine_cutter.clip()
                    logger(f"处理完成！求得区间：{section}")
                    final_range_g.append(section)
                    timelines.append(timeline)
                    current_task += 1
                    progressbar_callback(100)
                    time.sleep(.2)
                    self.update_taskprogress_label.emit(f"任务进度：{current_task}/{task_number}")
        except Exception as e:
            logger(f"出现错误：{e}")
            self.update_progressbar.emit(100)
            # raise e
            return
        logger("\n***********************************结果************************************\n")
        progressbar_callback(100)
        for frame_range,i in zip(final_range_g,timelines):
            m1, s1 = divmod(i[0] // 1000, 60)
            h1, m1 = divmod(m1, 60)
            m0, s0 = divmod(i[1] // 1000, 60)
            h0, m0 = divmod(m0, 60)
            logger(f"帧区间：{frame_range}, 时间轴：%02d:%02d:%02d  ~  %02d:%02d:%02d" % (h1, m1, s1, h0, m0, s0))
        logger("处理完成！")
        del fine_cutter
        
class concatenate_thread(cutter_thread):

    button_disable_sig = pyqtSignal(bool)
    load_finalpage_sig = pyqtSignal()
    
    def run(self):
        global final_range_g
        time.sleep(.05)
        self.button_disable_sig.emit(False)
        progressbar_callback = self.update_progressbar.emit
        logger = mpy_qt_progressbar(progressbar_callback)
        conc = concatenator(filepath_g,".\\tmp\\_result_.mp4",logger)
        conc.concatenate(final_range_g)
        self.update_progressbar.emit(100)
        time.sleep(.5)
        self.load_finalpage_sig.emit()
        self.button_disable_sig.emit(True)


class MyMainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self,parent =None):
        super(MyMainWindow,self).__init__(parent)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.openfile_button.clicked.connect(self.on_openfilebutton_clicked)
        self.next_0.clicked.connect(self.on_nextbutton_clicked)
        self.next_0.clicked.connect(self.crude_cut_proc)
        self.next_2.clicked.connect(self.on_nextbutton_clicked)
        self.next_2.clicked.connect(self.load_clip_selector)
        self.next_3.clicked.connect(self.on_nextbutton_clicked)
        self.next_3.clicked.connect(self.refine_cut_proc)
        # self.next_4.clicked.connect(self.on_nextbutton_clicked)
        # self.next_5.clicked.connect(self.on_nextbutton_clicked)
        self.term0_progressbar.setValue(0)
        self.crudecut_thread = crude_cutter_thread(self)
        self.crudecut_thread.update_textedit.connect(self.term0_textedit.append)
        self.crudecut_thread.update_progressbar.connect(self.term0_progressbar.setValue)
        template_list = os.listdir("templates")
        template_list = [i.split(".png")[0] for i in template_list if i.endswith(".png")]
        self.templateselector.addItems(template_list)
        self.templateselector.setCurrentIndex(0)
        self.templateselector.currentIndexChanged.connect(self.on_templateselector_changed)
        self.current_clip_selection = 0
        self.preview_2.setText("请先进行粗剪剪辑")
        self.position_silder.valueChanged.connect(self.on_position_silder_value_changed)
        self.lineEdit.textChanged.connect(self.on_pos_line_edit_changed)
        self.timetable_rowcount = 0
        self.frame_selection_button.clicked.connect(self.on_frameselectionbutton_clicked)
        self.nextvideo_toolbutton.clicked.connect(self.on_nextvideotoolbutton_clicked)
        self.lastvideo_toolbutton.clicked.connect(self.on_lastvideotoolbutton_clicked)
        self.refinecut_thread = refine_cutter_thread(self)
        self.refinecut_thread.update_textedit.connect(self.term1_textedit.append)
        self.refinecut_thread.update_progressbar.connect(self.term1_progressbar.setValue)
        self.refinecut_thread.update_taskprogress_label.connect(self.taskprogresslabel.setText)
        self.clip_widget_list = []
        self.clip_widget_label_layout_dict = {0:20,1:250,2:480}
        self.clip_widget_checkbox_layout_dict = {0:100,1:330,2:560}

        # global final_range_g
        # final_range_g.append((106973, 109023))
        # final_range_g.append((70863, 71763))
        # final_range_g.append((114514,919810))
        # self.create_clip_widgets()
        self.next_4.clicked.connect(self.create_clip_widgets)
        self.selection_progressbar.setValue(0)
        self.next_5.clicked.connect(self.filter_selected_clips_and_start_concatenate)
        self.conc_thread = concatenate_thread(self)
        self.conc_thread.update_progressbar.connect(self.selection_progressbar.setValue)
        # self.conc_t.update_textedit.connect(self.term5_textedit.append)
        self.conc_thread.button_disable_sig.connect(self.next_5.setEnabled)
        self.conc_thread.load_finalpage_sig.connect(self.load_export_page)
        self.outputselectionbutton.clicked.connect(self.on_exportpathbutton_clicked)
        self.exportbutton.clicked.connect(self.on_export_button_clicked)
        self.ok.clicked.connect(self.on_ok_button_clicked)
        self.tabWidget.setMovable(False)
        # self.tabWidget.setTabsClosable(False)
        
        self.default_export_path_POSIX = "~/Desktop/surgery_autoclipped.mp4"
        self.default_export_path_WIN32 = "..\\surgery_autoclipped.mp4"
        self.from_json("config.json")

    def from_json(self,json_filepath:str) -> None:
        try:
            with open(json_filepath,"r") as f:
                json_dict = json.load(f)["preference"]
            for key, value in json_dict.items():
                setattr(self, key, value)
        except Exception as e:
            print(f"读取配置文件失败：{e}\n使用默认配置")

    def read_frame_from_cap_g(self,pos:int,output_callback,error_callback = None) -> None:
        global cap_g
        cap_g.set(cv2.CAP_PROP_POS_FRAMES,pos)
        fl,fr = cap_g.read()
        if fl:
            fr = cv2.cvtColor(fr,cv2.COLOR_BGR2RGB,None)
            h,w,c = fr.shape
            qimg = QImage(fr.data,w,h,3*w,QImage.Format_RGB888)
            qpixmap=QPixmap.fromImage(qimg)
            output_callback(qpixmap)
        elif error_callback:
            error_callback("无法打开文件，可能已经删除或移动")

    def on_openfilebutton_clicked(self) -> None:
        global cap_g
        global filepath_g
        filename , filetype = QFileDialog.getOpenFileName(self)
        if filename:
            cap_g = cv2.VideoCapture(filename)
            filepath_g = filename
            self.file_path_label.setText( f"视频文件：{filename}")
            cap_g.set(cv2.CAP_PROP_POS_FRAMES,0)
            fl , fr = cap_g.read()
            if fl:
                fr = cv2.cvtColor(fr,cv2.COLOR_BGR2RGB,None)
                h,w,c = fr.shape
                qimg = QImage(fr.data,w,h,3*w,QImage.Format_RGB888)
                qpixmap=QPixmap.fromImage(qimg)
                self.preview.setPixmap(qpixmap)
            else:
                self.preview.setText("无法打开文件")
            # cap_g.release()
    
    def on_templateselector_changed(self) -> None:
        raise NotImplementedError

    def on_nextbutton_clicked(self) -> None:
        index = self.tabWidget.currentIndex()
        self.tabWidget.setCurrentIndex(index+1)

    def crude_cut_proc(self) -> None:
        global cap_g
        self.term0_textedit.clear()
        if cap_g is None:
            self.term0_textedit.append("请先打开文件")
            return

        self.term0_progressbar.setValue(0)
        self.crudecut_thread.start()
        
    def load_clip_selector(self) -> None:
        global cap_g
        global crude_range_g
        global key_frame_g
        self.current_clip_selection = 0
        key_frame_g = KeyFrameDict({0:[]})
        self.lastvideo_toolbutton.setEnabled(False)
        if len(crude_range_g) == 1:
            self.nextvideo_toolbutton.setEnabled(False)
        if crude_range_g is None:
            pass
            return
        else:
            self.read_frame_from_cap_g(crude_range_g[0][0],self.preview_2.setPixmap,self.preview_2.setText)
            self.lineEdit_2.setText(str(self.current_clip_selection+1))
            self.lineEdit.setText(str(crude_range_g[0][0]))
            self.position_silder.setMinimum(crude_range_g[0][0])
            self.position_silder.setMaximum(crude_range_g[0][1])
            self.position_silder.setValue(crude_range_g[0][0])
            self.timetable_rowcount = 0
            self.timelinetable.clearContents()
            self.label_2.setText(f"/{len(crude_range_g)}")
        # from lib.crude_cutter import CrudeCutter
        

    def on_position_silder_value_changed(self) -> None:
        global cap_g
        global crude_range_g
        pos = self.position_silder.value()
        self.read_frame_from_cap_g(pos,self.preview_2.setPixmap,self.preview_2.setText)
        self.lineEdit.setText(str(pos))


    def on_pos_line_edit_changed(self) -> None:
        global cap_g
        global crude_range_g
        try:
            pos = int(self.lineEdit.text())
            self.read_frame_from_cap_g(pos,self.preview_2.setPixmap,self.preview_2.setText)
            self.position_silder.setValue(pos)
        except:
            self.position_silder.setValue(0)

    # 绷不住了，pyqt居然还会自动连接，长难句启动！
    def on_frameselectionbutton_clicked(self) -> None:
        global cap_g
        global crude_range_g
        global key_frame_g
        pos = int(self.lineEdit.text())
        key_frame_g[self.current_clip_selection].append(pos)
        index = QTableWidgetItem(f"关键帧{self.timetable_rowcount}")
        pos = QTableWidgetItem(str(pos))
        self.timelinetable.setItem(self.timetable_rowcount,0,index)
        self.timelinetable.setItem(self.timetable_rowcount,1,pos)
        self.timetable_rowcount += 1

    def on_nextvideotoolbutton_clicked(self) -> None:
        global key_frame_g
        global crude_range_g
        global cap_g

        self.lastvideo_toolbutton.setEnabled(True)
        self.current_clip_selection += 1
        if self.current_clip_selection + 1 >= len(crude_range_g):
            self.nextvideo_toolbutton.setEnabled(False)

        self.read_frame_from_cap_g(crude_range_g[self.current_clip_selection][0],self.preview_2.setPixmap,self.preview_2.setText)
        self.lineEdit_2.setText(str(self.current_clip_selection+1))
        self.lineEdit.setText(str(crude_range_g[self.current_clip_selection][0]))
        self.position_silder.setMinimum(crude_range_g[self.current_clip_selection][0])
        self.position_silder.setMaximum(crude_range_g[self.current_clip_selection][1])
        self.position_silder.setValue(crude_range_g[self.current_clip_selection][0])
        self.timelinetable.clearContents()
        self.timetable_rowcount = 0
        try :
            keyframes = key_frame_g[self.current_clip_selection]
            for i in keyframes:
                index = QTableWidgetItem(f"关键帧{self.timetable_rowcount}")
                pos = QTableWidgetItem(str(i))
                self.timelinetable.setItem(self.timetable_rowcount,0,index)
                self.timelinetable.setItem(self.timetable_rowcount,1,pos)
                self.timetable_rowcount += 1
        except KeyError:
            key_frame_g[self.current_clip_selection] = []
            

    def on_lastvideotoolbutton_clicked(self) -> None:
        global key_frame_g
        global crude_range_g
        global cap_g

        self.nextvideo_toolbutton.setEnabled(True)
        self.current_clip_selection -= 1
        if self.current_clip_selection <= 0:
            self.lastvideo_toolbutton.setEnabled(False)

        self.read_frame_from_cap_g(crude_range_g[self.current_clip_selection][0],self.preview_2.setPixmap,self.preview_2.setText)
        self.lineEdit_2.setText(str(self.current_clip_selection+1))
        self.lineEdit.setText(str(crude_range_g[self.current_clip_selection][0]))
        self.position_silder.setMinimum(crude_range_g[self.current_clip_selection][0])
        self.position_silder.setMaximum(crude_range_g[self.current_clip_selection][1])
        self.position_silder.setValue(crude_range_g[self.current_clip_selection][0])
        self.timelinetable.clearContents()
        self.timetable_rowcount = 0
        try :
            keyframes = key_frame_g[self.current_clip_selection]
            for i in keyframes:
                index = QTableWidgetItem(f"关键帧{self.timetable_rowcount}")
                pos = QTableWidgetItem(str(i))
                self.timelinetable.setItem(self.timetable_rowcount,0,index)
                self.timelinetable.setItem(self.timetable_rowcount,1,pos)
                self.timetable_rowcount += 1
        except KeyError:
            key_frame_g[self.current_clip_selection] = []

    def refine_cut_proc(self) -> None:
        global cap_g
        global final_range_g
        final_range_g = []
        self.term1_textedit.clear()
        if cap_g is None:
            self.term1_textedit.append("请先打开文件")
            return

        self.term1_progressbar.setValue(0)
        self.refinecut_thread.start()

    def create_clip_widgets(self) -> None:
        global final_range_g
        if self.clip_widget_list:
            for i in self.clip_widget_list:
                i[0].deleteLater()
                i[1].deleteLater()
        self.clip_widget_list = []
        self.selection_progressbar.setValue(0)
        if len(final_range_g) == 0:
            return
        for clip in final_range_g:
            self.append_clip_widgets(clip)
        # 搞了半天，最后发现小组件不能在当前tab添加，一定要添加完再切换
        self.on_nextbutton_clicked()



    def append_clip_widgets(self,final_clip) -> None:
        global cap_g
        # if len(self.clip_widget_list) != 0:
        col = (len(self.clip_widget_list)) % 3
        row = len(self.clip_widget_list) //3
        # print(f"col:{col},row:{row}")


        label_x = self.clip_widget_label_layout_dict[col]
        checkbox_x = self.clip_widget_checkbox_layout_dict[col]

        calculate_label_y = lambda x : 20 + 210 * x
        calculate_checkbox_y = lambda x : 180 + 210 * x

        label_y = calculate_label_y(row)
        checkbox_y = calculate_checkbox_y(row)

        self.scrollArea_4.setWidgetResizable(True)

        newlabel = QtWidgets.QLabel(self.scrollAreaWidgetContents_5)
        newlabel.setEnabled(True)
        newlabel.setGeometry(QtCore.QRect(label_x, label_y, 200, 150))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(newlabel.sizePolicy().hasHeightForWidth())
        newlabel.setSizePolicy(sizePolicy)
        newlabel.setMinimumSize(QtCore.QSize(200, 150))
        newlabel.setMaximumSize(QtCore.QSize(200, 150))
        newlabel.setLayoutDirection(QtCore.Qt.LeftToRight)
        newlabel.setStyleSheet("background-color: rgb(191, 191, 191);")
        newlabel.setAlignment(QtCore.Qt.AlignCenter)
        newlabel.setObjectName(f"newlabel_{len(self.clip_widget_list)}")
        newlabel.setScaledContents(True)
        self.read_frame_from_cap_g((final_clip[1] + final_clip[0]) // 2,newlabel.setPixmap,newlabel.setText)

        newcheckBox = QtWidgets.QCheckBox(self.scrollAreaWidgetContents_5)
        newcheckBox.setGeometry(QtCore.QRect(checkbox_x, checkbox_y, 80, 20))
        newcheckBox.setObjectName(f"checkBox_{len(self.clip_widget_list)}")
        newcheckBox.setText("选择")
        newcheckBox.setChecked(True)
        

        # print(f"newlabel_{len(self.clip_widget_list)},x:{label_x},y:{label_y}")
        # print(f"checkBox_{len(self.clip_widget_list)},x:{checkbox_x},y:{checkbox_y}")
        # print(f"clip:{final_clip}")
        
        self.clip_widget_list.append((newlabel,newcheckBox))

    def filter_selected_clips_and_start_concatenate(self) -> None:
        global final_range_g
        selected_clips = []
        for i in range(len(self.clip_widget_list)):
            if self.clip_widget_list[i][1].isChecked():
                selected_clips.append(final_range_g[i])
        final_range_g = selected_clips
        self.conc_thread.start()

    def load_export_page(self) -> None:
        cap_final = cv2.VideoCapture(".\\tmp\\_result_.mp4")
        # cap_final.set(cv2.CAP_PROP_POS_FRAMES,0)
        fl,fr = cap_final.read()
        if fl:
            fr = cv2.cvtColor(fr,cv2.COLOR_BGR2RGB,None)
            h,w,c = fr.shape
            qimg = QImage(fr.data,w,h,3*w,QImage.Format_RGB888)
            qpixmap=QPixmap.fromImage(qimg)
            self.preview_3.setScaledContents(True)
            self.preview_3.setPixmap(qpixmap)
        
        self.outputpathlineEdit_3.setText(self.default_export_path_WIN32)
        self.on_nextbutton_clicked()
        
    def on_exportpathbutton_clicked(self) -> None:
        export_path = QFileDialog.getSaveFileName(self,"选择导出路径",".","*.mp4")
        if export_path[0]:
            self.outputpathlineEdit_3.setText(export_path[0])

    def on_export_button_clicked(self) -> None:
        os.system(f"move .\\tmp\\_result_.mp4 {self.outputpathlineEdit_3.text()}")
        # self.tabWidget.setCurrentIndex(0)
        self.exportbutton.setEnabled(False)

    def on_ok_button_clicked(self) -> None:
        try:
            os.system(f"move .\\tmp\\_result_.mp4 {self.outputpathlineEdit_3.text()}")
        except:
            pass
        self.close()

 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())    