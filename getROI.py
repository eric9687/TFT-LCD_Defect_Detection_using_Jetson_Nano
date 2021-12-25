import cv2
import os
import numpy as np

class getROI:
  def processImg(self,path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰白图处理
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    fgamma = 4 # 伽马变换，图像增强
    gray = np.uint8(np.power((np.array(gray) / 255.0), fgamma) * 255.0)  #np.power(a,b):a^b
    gray = cv2.convertScaleAbs(gray, alpha=1, beta=0)  # 增强对比度
    gray = cv2.normalize(gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)  #중복을 최소화=정규화# 进行归一化
    #gray = cv2.equalizeHist(gray)  # 直方图均衡化 #이미지의 contrast를 넓게 퍼뜨림(밝기를 조정)
    ret, th1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # 二值图处理
    # 开闭运算去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 返回指定形状和尺寸的结构元素（内核矩阵）
    # th1 = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)  # 开运算，先腐蚀后膨胀，消除小物体（白色)
    th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)  # 闭运算，排除小型黑洞（黑色区域)
    # canny(img,minVal,maxVal) 设置阈值  一般maxVal=3*minVal
    # 卷积核的大小默认为3
    edge = cv2.Canny(th1, 100, 300)

    return th1, img, edge
   # 利用轮廓处理获取roi
  def getRoi(self,img, binary):
    '''
    img: 原图
    binary: 预处理后得到的canny边缘
    '''
    # 寻找轮廓，输出提取轮廓的数目
    contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_list = []

    # 判断MURA缺陷
    for cnt in range(len(contours)):
        area = cv2.contourArea(contours[cnt])
        # 判断提取所需的轮廓，经验值需要调试获取
        if 10 < area < 10000:
            # 获取外接矩形的值
            x, y, w, h = cv2.boundingRect(contours[cnt])
            roi_list.append((x, y, w, h))  # 在数组后加上相应的元素
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img, roi_list

  def classify(self,root_path,root_path1):  #分别保存正样本和负样本ROI

          dir = root_path + "\\"
          for root, dir, files in os.walk(dir):
              for file in files:
                  th1, img, edge = getROI().processImg(root_path + "\\" + str(file))
                  # copy img
                  src = img.copy()
                  # 获取roi
                  img, roi_list = getROI().getRoi(img, edge)
                  for i in range(len(roi_list)):
                      x, y, w, h = roi_list[i]
                      roi = src[y:y + h, x:x + w]
                      # cv2.imshow("roi image", img)
                      cv2.imwrite("hyun_negative_data\\roii_%d" % i + str(file), roi)

          dir1 = root_path1 + "\\"
          for root, dir1, files in os.walk(dir1):
              for file in files:
                  th1, img, edge = getROI().processImg(root_path1 + "\\" + str(file))
                  # copy img
                  src = img.copy()
                  # 获取roi
                  img, roi_list = getROI().getRoi(img, edge)
                  for i in range(len(roi_list)):
                      x, y, w, h = roi_list[i]
                      roi = src[y:y + h, x:x + w]
                      # cv2.imshow("roi image", img)
                      cv2.imwrite("hyun_positive_data\\roii_%d" % i + str(file), roi)