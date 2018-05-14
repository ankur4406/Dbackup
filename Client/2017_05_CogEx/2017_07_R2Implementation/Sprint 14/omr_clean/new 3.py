"""

Purpose: Extract bounding boxes for any incoming new form to support templating UI
Input(s): 1. Instance of a unknown form type as a .png file

Outputs(s): 1. A configuration file (.json) for each element with a coordinate map particular to the form instance
			2. Individual slices stored as separate png files

Created by: Nidhi Purohit & Denver Dias, Deloitte USI
Created on: October, 6, 2017
Modified by:

"""
import numpy as np
import argparse
# import imutils
import glob
import cv2
import json
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import os

def extract_form_layout(filename):

    #reading the image
    form_image = cv2.imread(filename)

    #convert to gray scale
    gray_image = cv2.cvtColor(form_image, cv2.COLOR_BGR2GRAY)
    print(gray_image.shape)

    #inverting the image pixels
    img_invert = cv2.bitwise_not(gray_image)

    #Applying Blur/Filter
    # blur = cv2.GaussianBlur(test_form_1,(1,1),1000)
    blur = cv2.GaussianBlur(img_invert, (5, 5), 0)
    cv2.imwrite('image_blur.png', blur)

    #Applying Thresholding
    # ret, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    # th2 = cv2.adaptiveThreshold(test_form_1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\cv2.THRESH_BINARY,11,2)
    # th3 = cv2.adaptiveThreshold(test_form_1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\cv2.THRESH_BINARY,11,2)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('image_thresh.png', thresh)

    #Dilation & Erosion
    # kernel = np.ones((5,5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_dilation = cv2.dilate(thresh, kernel, iterations=3)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    cv2.imwrite('image_dilated.png', img_dilation)
    cv2.imwrite('image_erode.png', img_erosion)

    # Inverting the pixels
    img_inv = cv2.bitwise_not(img_erosion)
    cv2.imwrite('image_inverted.png', img_inv)

    #finding contours
    image, contours, hierarchy = cv2.findContours(img_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = form_image.copy()
    img = cv2.drawContours(img, contours, -1, (0, 0, 255), 4)
    cv2.imwrite('image_contours.png', img)

    print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Select long perimeters only
    perimeters = [cv2.arcLength(contours[i], True) for i in range(len(contours))]
    #cnts = [i for i in range(len(contours)) if perimeters[i] > perimeters[0] / 20]
    #cnts = [i for i in range(len(contours)) if cv2.boundingRect(contours[i])[2] > 0
            #and cv2.boundingRect(contours[i])[3] > 0]

    cnts = []

    for i in range(len(contours)):
        if cv2.boundingRect(contours[i])[2] > 80 and cv2.boundingRect(contours[i])[3] > 80:
            cnts.append(i)
    print(len(cnts))

    # Show image
    img_contour = form_image.copy()
    [cv2.drawContours(img_contour, [contours[i]], 0, (0, 255, 0), 5) for i in cnts]
    cv2.imwrite('image_ref_contours.png', img_contour)

    #storage_path = r'C:\Users\ddias\Documents\Cog Ex Project\Contour Detection\CMS_Output'
    storage_path = r'C:\Users\ddias\Documents\Cog Ex Project\Contour Detection\CSF_Output'

    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    #os.chdir(r'C:\Users\ddias\Documents\Cog Ex Project\Contour Detection\CMS_Output')
    os.chdir(r'C:\Users\ddias\Documents\Cog Ex Project\Contour Detection\CSF_Output')

    coordinates = {}

    for i in cnts:
        [x, y, w, h] = cv2.boundingRect(contours[i])

        cv2.imwrite(str(i) + ".png", form_image[y:y + h, x:x + w])

        coordinates[i] = [y, y + h, x, x + w]

        i = i + 1

    with open('coordinates.json', 'w') as fp:
        json.dump(coordinates, fp)



if __name__ == '__main__':

    extract_form_layout('merged_csf.png')
