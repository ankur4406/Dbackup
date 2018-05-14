# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 13:38:20 2017

@author: nipurohit
"""

import cv2
import numpy as np
import time,sys
import pylab as pl

# define range of  color in HSV
lower_red = np.array([150,135,160])
upper_red = np.array([180,250,200])
white_line  = np.array([255,255,255])
red_line   = np.array([0,0,255])

#############  Function to put vertices in clockwise order ######################
def rectify(h):
		''' this function put vertices of square we got, in clockwise order '''
		h = h.reshape((4,2))
		hnew = np.zeros((4,2),dtype = np.float32)

		add = h.sum(1)
		hnew[0] = h[np.argmin(add)]
		hnew[2] = h[np.argmax(add)]
		
		diff = np.diff(h,axis = 1)
		hnew[1] = h[np.argmin(diff)]
		hnew[3] = h[np.argmax(diff)]

		return hnew
		
#read image

im = cv2.imread('images\cropped_table_Frasier.png')

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray,(5,5),0)

cv2.imshow('imgray', im)
cv2.waitKey(0)

# Values below 150 goes to 0 (black, everything above goes to 255 (white)
ret,thresh = cv2.threshold(imgray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('1 Threshold Binary', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('CMS1500_FrasierClean_no grey.jpg', thresh)
'''
#ret,thresh = cv2.threshold(imgray,127,255,0)
thresh = cv2.adaptiveThreshold(imgray,255,1,1,11,2)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
'''
# Remove some small noise if any.
dilate = cv2.dilate(thresh,None)
cv2.imshow('dilate', dilate)
cv2.waitKey(0)
erode = cv2.erode(dilate,None)
cv2.imshow('erode', erode)
cv2.waitKey(0)
#cv2.imwrite('CMS1500_FrasierClean_no grey_erode.jpg', erode)

_,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##for contour in contours:
##    if (cv2.contourArea(contour) > 50):  
##        print contours
#im2 =  np.zeros((606,2301,3), np.uint8)
im2 =  np.zeros((618,2388,3), np.uint8)
cv2.drawContours(im2,contours,-1,(125,125,0),1)
cv2.imshow('contours',im2)
cv2.imwrite('frasiergray2.png',im2)
cv2.waitKey(0)

image_area = imgray.size	# this is area of the image

for i in contours:
	if cv2.contourArea(i)> image_area/2: # if area of box > half of image area, it is possibly the biggest blob
		peri = cv2.arcLength(i,True)
		approx = cv2.approxPolyDP(i,0.02*peri,True)
		#cv2.drawContours(img,[approx],0,(0,255,0),2,cv2.CV_AA)
		break

h = np.array([ [0,0],[2380,0],[2380,600],[0,600] ],np.float32)	# this is corners of new square image taken in CW order
#h = np.array([ [0,0],[2280,0],[2280,588],[0,588] ],np.float32)
approx=rectify(approx)	# we put the corners of biggest square in CW order to match with h

retval = cv2.getPerspectiveTransform(approx,h)	# apply perspective transformation
#warp = cv2.warpPerspective(im,retval,(2281,589))  # Now we get perfect square with size 450x450
warp = cv2.warpPerspective(im,retval,(2381,601))
warpg = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)	# kept a gray-scale copy of warp for further use
cv2.imwrite('frasierwarpgray2.png',warpg)

'''
#size of the image (height, width)
h, w = im.shape[:2]

#copy the original image to show the posible candidate
table_candidates = im.copy()

#biggest rectangle
size_rectangle_max = 0; 
for i in range(len(contours)):
    #aproximate countours to polygons
    approximation = cv2.approxPolyDP(contours[i], 4, True)
        
    #has the polygon 4 sides?
    if(not (len (approximation)==4)):
        continue;
    #is the polygon convex ?
    if(not cv2.isContourConvex(approximation) ):
        continue; 
    #area of the polygon
    size_rectangle = cv2.contourArea(approximation)
    #store the biggest
    if size_rectangle> size_rectangle_max:
        size_rectangle_max = size_rectangle 
        big_rectangle = approximation

#show the best candidate
approximation = big_rectangle
for i in range(len(approximation)):
    cv2.line(table_candidates,
             (big_rectangle[(i%4)][0][0], big_rectangle[(i%4)][0][1]), 
             (big_rectangle[((i+1)%4)][0][0], big_rectangle[((i+1)%4)][0][1]),
             (255, 0, 0), 2)
#show image
_=pl.imshow(table_candidates, cmap=pl.gray()) 
_=pl.axis("off")
cv2.imwrite('C:\D_CogX\SkewImage\images\candidate_crop.png',table_candidates)
'''