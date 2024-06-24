import cv2
import numpy as np
import pandas as pd
from google.colab.patches import cv2_imshow
from statistics import mean
import tensorflow as tf

#read image
image = cv2.imread('test.jpg')

#convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2_imshow(image)

#range limits
#defines ranges to split between black and white
#the higher the value the more sensitive
lower = np.array([0, 0, 120])
upper = np.array([0, 0, 255])
msk = cv2.inRange(image, lower, upper)
#turns black to white and vice versa
msk = cv2.bitwise_not(msk)
cv2_imshow(msk)

#it is necessary to define a kernel, in this case it is the shape of the figure we want to extract (rectangle, elipse, circle, etc). In this case rectangle
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

#erode the whites to remove the noise points, or at least reduce them
rrmsk = cv2.erode(msk,kernel,iterations = 1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,5))
rrmsk = cv2.dilate(msk, kernel, iterations=1)
cv2_imshow(rrmsk)

#sample: contours
contours, hierarchy = cv2.findContours(rrmsk, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
minw = 50
minh = 10
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cleancontours = []
words = []
for contour in contours:
 x,y,w,h = cv2.boundingRect(contour)
 if ((w>=minw) & (h>=minh)):
  cleancontours.append(contour)
  word = image[y-5:y+h+5,x-5:x+w+5]
  words.append(word)
cleancontours = tuple(cleancontours)
imageC = cv2.imread('test.jpg')
imageC = cv2.cvtColor(imageC, cv2.COLOR_BGR2HSV)
cv2.drawContours(imageC, cleancontours, -1, (0,255,0), 3)
cv2_imshow(imageC)

