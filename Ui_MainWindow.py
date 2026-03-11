
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import json

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from Ui_segmentation import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton#事件需要的引用
from PyQt5.QtCore import Qt, QDateTime,QTimer, QRect

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor

from eval_hardnet_OS import init_model, match_images

 
from eval import mse

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.sarbtn.clicked.connect(self.load_sar_img)  #按键触发事件
        self.ui.optbtn.clicked.connect(self.load_opt_img)  #按键触发事件
        #self.ui.vggnetrecobtn.clicked.connect(self.vggnet_reco_img)
        self.ui.rebtn.clicked.connect(self.resnet_reco_img) #按键触发事件

        self.ui.sarlabel.setStyleSheet("QLabel { background-color: #f8f8f8; border-width: 2px; border-style: solid; border-color: #4B91E5; }") #设置背景颜色和边框
        self.ui.optlabel.setStyleSheet("QLabel { background-color: #f8f8f8; border-width: 2px; border-style: solid; border-color: #4B91E5; }") #    设置背景颜色和边框    
        self.ui.reslabel.setStyleSheet("QLabel { background-color: #f8f8f8; border-width: 2px; border-style: solid; border-color: #4B91E5; }") #设置背景颜色和边框

        ## init model
        self.init_regi_model() #初始化模型

    def init_regi_model(self):
        self.embede_model, ldm_model, self.fp_detector = init_model() #初始化模型

    def resnet_reco_img(self):
        sar_img = self.qpix2numpy(self.ui.sarlabel.pixmap()) #sar图像
        opt_img = self.qpix2numpy(self.ui.optlabel.pixmap()) ##opt图像
        print(sar_img.shape,opt_img.shape)
        src_pts, dst_pts = match_images(sar_img, opt_img, self.embede_model, self.fp_detector, thresh=7,knn=2) #特征匹配
        opt_img = cv2.copyMakeBorder(opt_img, 0, 32, 0, 32, cv2.BORDER_CONSTANT, value=(128, 128, 128)) #添加边框
        opt_img = cv2.cvtColor(opt_img, cv2.COLOR_RGB2BGR) #转换颜色格式
        out_img = np.concatenate((sar_img, opt_img), axis=1) #拼接图像
        H,mask = cv2.findHomography(src_pts,dst_pts,cv2.USAC_MAGSAC,ransacReprojThreshold=0.25,
                                           confidence=0.99999, maxIters=10000)
        mse_src_pts = src_pts[mask.astype(bool).squeeze()] #    过滤掉不符合单应性矩阵的点
        mse_dst_pts = dst_pts[mask.astype(bool).squeeze()] # 过滤掉不符合单应性矩阵的点
        xmse, ymse, xymse, rate = mse(mse_src_pts, mse_dst_pts+32, 10) #计算均方误差

        mse_text = f"MSE: {round(xymse, 4)}"
        self.ui.mse_label.setText(mse_text)
        # cv2.putText(out_img, f"MSE:{round(xymse, 4)}, NCM:{rate}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) #添加文本
        H,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,ransacReprojThreshold=16) #计算单应性矩阵
        # cv2.putText(out_img, f"Match:{len(src_pts[mask.astype(bool).squeeze()])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) #添加文本
        max_lines = 50  # 设置最大线条数，如果存在无效线条可能会少于50
        valid_matches = np.where(mask)[0]  # 获取有效匹配点的索引
        step = max(1, len(valid_matches) // max_lines)  # 计算步长，确保最小步长为 1

        for i in range(0, len(valid_matches), step):
            if int(mask[valid_matches[i]]):  # 只绘制有效匹配点
                match_idx = valid_matches[i]
                # 获取源图像和目标图像的坐标
                src_x, src_y = int(src_pts[match_idx, 0]), int(src_pts[match_idx, 1])
                dst_x, dst_y = int(dst_pts[match_idx, 0]) + 512, int(dst_pts[match_idx, 1])

                # 在源图像和目标图像上绘制点
                cv2.circle(out_img, (src_x, src_y), 2, (255, 255, 0), -1)  # 绘制源点（黄色）
                cv2.circle(out_img, (dst_x, dst_y), 2, (255, 255, 0), -1)  # 绘制目标点（黄色）

                # 绘制源点和目标点之间的连接线（黄色）
                cv2.line(out_img, (src_x, src_y), (dst_x, dst_y), (0, 255, 0), 1)  # 绘制绿色线条
        pixmap = self.numpy2qpix(out_img) ##转换为QPixmap格式
        pixmap = pixmap.scaled(QtCore.QSize(1024, 512), QtCore.Qt.KeepAspectRatio) ##缩放图像
        self.ui.reslabel.setPixmap(pixmap) #设置图像
        

    def load_sar_img(self):      #按键触发事件
        """load_sar_img"""
        image_path, filetype = QFileDialog.getOpenFileName(self,
                  "选取文件",
                  "./",
                  "*png;;*jpg")  #设置文件扩展名过滤,注意用双分号间隔
        if image_path == '':
            return
        im = np.array(Image.open(image_path).convert('RGB'))
        pixmap = self.numpy2qpix(im)
        pixmap = pixmap.scaled(QtCore.QSize(512, 512), QtCore.Qt.KeepAspectRatio)
        self.ui.sarlabel.setPixmap(pixmap)
        self.image_path = image_path
        print(image_path,filetype)
    
    def load_opt_img(self):      #按键触发事件
        """load_opt_img"""
        image_path, filetype = QFileDialog.getOpenFileName(self,
                  "选取文件",
                  "./",
                  "*png;;*jpg")  #设置文件扩展名过滤,注意用双分号间隔
        if image_path == '':
            return
        im = np.array(Image.open(image_path).convert('RGB')) #读取图像
        pixmap = self.numpy2qpix(im) #转换为QPixmap格式
        pixmap = pixmap.scaled(QtCore.QSize(480, 480), QtCore.Qt.KeepAspectRatio) ##缩放图像
        self.ui.optlabel.setPixmap(pixmap) #设置图像
        self.image_path = image_path
        print(image_path,filetype)


    def qpix2numpy(self, pixmap):
        """return is BGR"""
        qimg = pixmap.toImage() #转换为QImage格式
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth()) #获取图像的高度和宽度
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]
        return result

    def numpy2qpix(self, image):
        """convert numpy to qpixmap"""
        im = Image.fromarray(image.astype(np.uint8)).convert('RGB')
        img = im.toqpixmap()
        return img

    def numpy2qpix(self, image):
        """convert numpy to qpixmap"""
        Y, X = image.shape[:2]
        _bgra = np.zeros((Y, X, 3), dtype=np.uint8, order='C')
        _bgra[..., 0] = image[..., 0]
        _bgra[..., 1] = image[..., 1]
        _bgra[..., 2] = image[..., 2]
        qimage = QtGui.QImage(_bgra.data, X, Y, 3 * X, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        return pixmap
    

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    MainWindow = MainWindow()  # 创建主窗口
    #ui = UI_MainWindow()
    #ui.setupUi(MainWindow)
    #MainWindow.setStyleSheet("#MainWindow{background-image: url(bg.jpeg)}")
    MainWindow.show()  # 显示主窗口
    sys.exit(app.exec_())  #
