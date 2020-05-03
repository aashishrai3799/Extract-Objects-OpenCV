import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

pathin = 'C:/Users/server_pc_redhat/Documents/drapezy/data/unpreprocessed/'
pathout = 'C:/Users/server_pc_redhat/Documents/drapezy/data/processed/'
images = os.listdir(pathin)
 
img = cv2.imread(pathin + 'pic13.jpg')

org_img = img
img = cv2.resize(img, (2500, 1900), interpolation = cv2.INTER_AREA)
img = img[10:,:-10]
#cv2.imshow('image', img)
image = img
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

'''
#blurred = gray
canny = cv2.Canny(blurred, 150, 250, 100)
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=1)
'''
 
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

cv2.imshow("Edges", np.hstack([wide, tight, auto]))
#cv2.waitKey(0)
kernel = np.ones((5,5), np.uint8)

dilate = cv2.dilate(auto, kernel, iterations=1)
# Find contours
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#print(cnts)
# Iterate thorugh contours and filter for ROI
areas = [cv2.contourArea(c) for c in cnts]
max_index = np.argmax(areas)
cnt=cnts[max_index]

c = cnt
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.02 * peri, True)
cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

x,y,w,h = cv2.boundingRect(cnt)
print(x,y,w,h)
image2 = image[(y+10):(y+h-10), (x+10):(x+w-10)]
image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

#cv2.imshow('canny', canny)
#cv2.imwrite(pathout + 'pic2.jpg', image2)

image2 = cv2.resize(image2, (625, 475), interpolation = cv2.INTER_AREA)
image = cv2.resize(image, (625, 475), interpolation = cv2.INTER_AREA)
#pathout = cv2.resize(pathout, (625, 475), interpolation = cv2.INTER_AREA)

cv2.imshow('cropped', image2)
cv2.imshow('final', image)



cv2.waitKey(0)
