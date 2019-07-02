import cv2

frame = cv2.imread('IMG_42.jpg') # origin image
heatmap = cv2.imread('1.jpg') # heatmap image
x, y = frame.shape[0:2]
heatmap = cv2.resize(heatmap, (y, x))
overlay = frame.copy()
alpha = 0.6 # 设置覆盖图片的透明度
cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1) # 设置蓝色为热度图基本色
cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame) # 将背景热度图覆盖到原图
cv2.addWeighted(heatmap, alpha, frame, 1-alpha, 0, frame) # 将热度图覆盖到原图
cv2.imshow('frame', frame)
cv2.waitKey(0)