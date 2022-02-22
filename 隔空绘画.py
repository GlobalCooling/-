import cv2 as cv
import numpy as np
import pygame

pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("这是一块神奇的画板 幽蓝伊梦")
cap = cv.VideoCapture(0)
while True:
    #读取一帧
    _, frame = cap.read()
    #左右翻转，不同的摄像头可能不需要翻转
    frame=cv.flip(frame,1)
    #将BGR转换到HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #识别的颜色范围
    lower_color = np.array([170,100,100])
    upper_color = np.array([180,255,255])
    #构建掩膜
    mask = cv.inRange(hsv, lower_color, upper_color)
    #开运算去除噪点
    mask = cv.erode(mask, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=1)
    #按位与 掩膜和原图
    cv.bitwise_and(frame,frame, mask= mask)
    #提取轮廓，注意opencv版本，3.x版本的findContours函数返回结果是三个
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    #如果有轮廓
    if len(cnts)>0:
        #选出最大的轮廓
        cnt=max(cnts, key=cv.contourArea)
        #绘制轮廓
        #res=cv.drawContours(res.copy(),cnts,-1,(0,255,0),3)
        #绘制边界矩形
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #计算轮廓矩M和质心坐标center
        M=cv.moments(cnt)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        #在摄像头画面和pygame里绘制质心
        cv.circle(frame,center,3,(0,255,0),-1)
        pygame.draw.circle(screen,(255,255,0),center,5)
        pygame.display.update()

    cv.imshow('frame',frame)
    #cv.imshow('mask',mask)
    #如果按下esc键就退出
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
#释放摄像头，关闭窗口
cap.release()
cv.destroyAllWindows()
#保存图片
pygame.image.save(screen,"draw.png")
print("图片draw.png已保存")
pygame.quit()