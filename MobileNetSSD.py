# coding = utf-8
import numpy as np 
import cv2
import time
import argparse

def get_argement():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default="00001.jpg", help="image")
    ap.add_argument("-c", "--confidence", default=0.2,
                    help="minimun probability to fliter")
    return ap

args = get_argement()
image_path = "00001.jpg"
start = time.time()

# 载入模型 
# windows下建议powershell运行脚本，不然可能会有无法打开deploy file 的尴尬
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_15000.caffemodel")

image = cv2.imread(image_path)
(h, w) = image.shape[:2]
ima = cv2.resize(image, (300, 300))
# write to blob
blob = cv2.dnn.blobFromImage(ima, 0.007843, (300,300), 127.5)

# net forward
net.setInput(blob)
# 比如caffe模块的net.data.data无法实现
detections = net.forward()

# 查看网络输出的最后一层，第三维表示predict的box个数，第四维分别[0,label,confidence,box]
print(detections)
print(detections.shape)


for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	if confidence > 0.2:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
		idx = int(detections[0, 0, i, 1])
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		
		(startX, startY, endX, endY) = box.astype("int")

		# display the prediction
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(255,0,0), 2)

# # show the output image
#cv2.imshow("Output", image)

cost = time.time() - start
print("cost {}s".format(cost))
#cv2.waitKey(0)
