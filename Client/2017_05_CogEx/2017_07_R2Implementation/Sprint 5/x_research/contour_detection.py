
import cv2
import numpy as np
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

_,contours,heirarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #use RETR_EXTERNAL as mode
##for contour in contours:
##    if (cv2.contourArea(contour) > 50):  
##        print contours
#im2 =  np.zeros((606,2301,3), np.uint8)
im2 =  np.zeros((618,2388,3), np.uint8)
cv2.drawContours(im2,contours,-1,(125,125,0),1)
cv2.imshow('contours',im2)
cv2.imwrite('frasiergray2.png',im2)
cv2.waitKey(0)