"""

Purpose: To create an algorith for parsing text from an unstructured scanned document (image)
Input(s): 1. Scanned image document
          2. Pre-trained classification model for document body identification
Outputs(s): A json file in a pre-defined format containing the converted text along with other requisite information
Created by: Ankur Arora, Deloitte USI
Created on: September, 27, 2017
Modified by: 
Modified on:

"""

# Setting up by importing all the relevant libraries
import os
import sys
import cv2
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import pytesseract
from PIL import Image
import json
from scipy import misc

def extract_image_text(filename):

    # Import the image as a numpy array
    image = cv2.imread(filename,0)
    img = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Slice the image into sections based on whitespace
    # 1. Remove the white borders, if any
    p_image = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(p_image > 0))
    x,y,w,h = cv2.boundingRect(coords)
    image_sliced = image[np.max((x-10,0)) : np.min((x+w+10,image.shape[0])),
                         np.max((y-10,0)) : np.min((y+h+10, image.shape[1]))]

    # 2. Find lines by horizontally blurring the image and thresholding
    blur = cv2.blur(image_sliced, (91,9))
    b_mean = np.mean(blur, axis=1)/255
    threshold = np.percentile(b_mean, 50)
    t = b_mean > threshold
    byte_lines = np.where(1-t)
    byte_lines = byte_lines[0]
    
    # 3. Calculate the median linespace value for defining sections 
    linspace = []
    for x in range(byte_lines.shape[0]-1):
        if byte_lines[x+1] == byte_lines[x] + 1:
            continue
        linspace.append(byte_lines[x+1]-byte_lines[x]-1)
        linspace_limit = (1.0 * np.median(linspace))

    # 4. Add in extra byte lines to cover unwanted linespace
    for x in range(byte_lines.shape[0]-1):
        if byte_lines[x+1] == byte_lines[x] + 1:
            continue
        if ((byte_lines[x+1]-byte_lines[x]) <= linspace_limit):
            for i in range(byte_lines[x+1]-byte_lines[x]-1):
                byte_lines = np.append(byte_lines, (byte_lines[x]+i+1))
    byte_lines = np.sort(byte_lines)

    # 5. Identify text line coordinates (y) based on byte lines
    txt_lines_y = []
    start_y = byte_lines[0]
    for y in range(1, byte_lines.shape[0]-1):
        if byte_lines[y] == byte_lines[y-1] + 1:
            continue
        # identified gap between lines, close previous line and start a new one
        end_y = byte_lines[y-1]
        txt_lines_y.append([start_y, end_y])
        start_y = byte_lines[y]
    end_y = byte_lines[-1]
    txt_lines_y.append([start_y, end_y])

    # 6. Identify text line coordinates (x) based on non blank columns
    txt_lines_x = []
    for line in txt_lines_y:
        xx = []
        for x in range(image_sliced.shape[1]):
            col = image_sliced[line[0]:line[1], x]
            if np.min(col) < 128:
                xx.append(x)
        txt_lines_x.append([min(xx), max(xx)])

    # 7. Slice the document based on the coordinates and create features for type classification
    feature_list = ['width','height','seq_from_top','seq_from_bottom','width_prv',
                    'height_prv','width_nxt','height_nxt']

    features = pd.DataFrame(index=[], columns=feature_list)
    features.index.name = "filename"

    for i in range(len(txt_lines_x)):
        slc = image_sliced[np.max((txt_lines_y[i][0] - 2,0)) : np.min((txt_lines_y[i][1] + 2,image.shape[0])), 
                           np.max((txt_lines_x[i][0] - 2,0)) : np.min((txt_lines_x[i][1] + 2,image.shape[1]))]
        filename1 = os.path.splitext(filename)[0] + '_slice' + str(i+1) + os.path.splitext(filename)[1]
        misc.imsave(filename1, slc)
        features.loc[filename1] = [(txt_lines_x[i][1] - txt_lines_x[i][0] + 4)/image_sliced.shape[1],
                                  (txt_lines_y[i][1] - txt_lines_y[i][0] + 4)/image_sliced.shape[0],
                                  (i+1),
                                  (len(txt_lines_y)-i),
                                  0 if not i else (txt_lines_x[i-1][1] - txt_lines_x[i-1][0] + 4)/image_sliced.shape[1],
                                  0 if not i else (txt_lines_y[i-1][1] - txt_lines_y[i-1][0] + 4)/image_sliced.shape[0],
                                  0 if i==len(txt_lines_x)-1 else (txt_lines_x[i+1][1] - txt_lines_x[i+1][0] + 4)/image_sliced.shape[1],
                                  0 if i==len(txt_lines_y)-1 else (txt_lines_y[i+1][1] - txt_lines_y[i+1][0] + 4)/image_sliced.shape[0],
                                 ]

    # Import the pre trained model and classify sections
    model_classify = joblib.load('section_classifier.pkl')
    features['prediction'] = model_classify.predict(features)
    
    # Run an OCR and extract text from the requisite slices
    ocr = []
    cnt = 0
    for key, row in features.iterrows():
        if row['prediction']:
            out = {
                    "index": cnt,
                    "section": "",
                    "attribute": "", 
                    "text": pytesseract.image_to_string(Image.open(key)),
                    "type": "text"
            }
            cnt = cnt + 1
            ocr.append(out)
        os.remove(key)

    # Format the output in pre-defined format
    img_content = {
                    "data": ocr,
                    "attachment_type": "image", 
                    "type": "attachment"
                  } 
    
    # Dump the final output as a json
    with open(os.path.splitext(filename)[0] + ".json", 'w') as fp:
        json.dump(img_content, fp, indent=4)
        
# Define the main function for standalone script call
if __name__ == '__main__':
    extract_image_text(sys.argv[1])