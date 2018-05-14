"""
    Purpose: To split the unstructured letter documents based on whitespace; classify the sections based on a set of rules
                and perform an OCR for each section. Consolidate and store the results in a text file.
    Input(s): Unstructured letter document in the form of an image file (.jpg)
    Outputs(s): Parsed letter document saved as a text file (.txt) with the same name as the input file
    Created by: Ankur Arora, Deloitte USI
    Created on: August, 17, 2017
    Modified by: 
    Modified on: 
"""

# Setting up by importing all the relevant libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import misc
import numpy as np
import pytesseract
from PIL import Image

def read_letters(image_file):

    print ("Parsing " + image_file + " ...")
    
    # Loading the form instance as an array & perform the required preprocessing
    image = cv2.imread(image_file,0)
    image = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Slice the document to remove the extra whitespace borders, if any
    p_image = cv2.bitwise_not(image)
    p_image = cv2.threshold(p_image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(p_image > 0))
    x,y,w,h = cv2.boundingRect(coords)
    image_sliced = image[x-10:x+w+10,y-10:y+h+10]
    
    # Find lines by horizontally blurring the image and thresholding
    blur = cv2.blur(image_sliced, (91,9))
    b_mean = np.mean(blur, axis=1)/255
    threshold = np.percentile(b_mean, 66)
    t = b_mean > threshold
    byte_lines = np.where(1-t)
    byte_lines = byte_lines[0]
    
    # Calculate the median linespace value for defining sections 
    linspace = []
    for x in range(byte_lines.shape[0]-1):
        if byte_lines[x+1] == byte_lines[x] + 1:
            continue
        linspace.append(byte_lines[x+1]-byte_lines[x]-1)
    linspace_limit = (1.5 * np.median(linspace))
    
    # Add in extra byte lines to cover unwanted linespace
    for x in range(byte_lines.shape[0]-1):
        if byte_lines[x+1] == byte_lines[x] + 1:
            continue
        if ((byte_lines[x+1]-byte_lines[x]) <= linspace_limit):
            for i in range(byte_lines[x+1]-byte_lines[x]-1):
                byte_lines = np.append(byte_lines, (byte_lines[x]+i+1))
    byte_lines = np.sort(byte_lines)
    
    # Identify text line coordinates (y) based on byte lines
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

    # Identify text line coordinates (x) based on non blank columns
    txt_lines_x = []
    for line in txt_lines_y:
        xx = []
        for x in range(image_sliced.shape[1]):
            col = image_sliced[line[0]:line[1], x]
            if np.min(col) < 128:
                xx.append(x)
        txt_lines_x.append([min(xx), max(xx)])
    
    # Slice the document based on the coordinates and perform OCR. 
    # Also, create basic features for use in learning in later versions.
    features = []
    for i in range(len(txt_lines_x)):
        slc = image_sliced[txt_lines_y[i][0] - 2:txt_lines_y[i][1] + 2, 
                           txt_lines_x[i][0] - 2:txt_lines_x[i][1] + 2]
        filename = 'slice_' + str(i+1) + '.png'
        misc.imsave(filename, slc)
        ocr_txt = pytesseract.image_to_string(Image.open(filename))
        os.remove(filename)
        features.append({'flag': filename,
                        'width': (txt_lines_x[i][1] - txt_lines_x[i][0] + 4),
                        'heigth': (txt_lines_y[i][1] - txt_lines_y[i][0] + 4),
                        'seq_from_top': (i+1),
                        'seq_from_bottom': (len(txt_lines_y)-i),
                        'width_prv': 0 if not i else (txt_lines_x[i-1][1] - txt_lines_x[i-1][0] + 4),
                        'heigth_prv': 0 if not i else (txt_lines_y[i-1][1] - txt_lines_y[i-1][0] + 4),
                        'width_nxt': 0 if i==len(txt_lines_x)-1 else (txt_lines_x[i+1][1] - txt_lines_x[i+1][0] + 4),
                        'heigth_nxt': 0 if i==len(txt_lines_y)-1 else (txt_lines_y[i+1][1] - txt_lines_y[i+1][0] + 4),
                        'text': ocr_txt
                        })    
    
    # Do a section classification - rule based (to be updated with a learning algo in later versions)
    for i in range(len(features)):
        # Rule 1: First section is always a 'header'
        if features[i]['seq_from_top'] == 1:
            features[i]['type'] = 'header';
        # Rule 2: Any subsequent section with width less than 50% of the page width is a 'header'
        elif ((features[i]['width'] < (0.5 * image_sliced.shape[1])) & (features[i-1]['type'] == 'header')):
            features[i]['type'] = 'header'
        # Rule 3: Last section is a 'footer' only if the width is less than 50% of the page width
        elif ((features[i]['seq_from_bottom'] == 1) & (features[i]['width'] < (0.5 * image_sliced.shape[1]))):
            features[i]['type'] = 'footer'
        # Rule 4: All other sections are letter 'body' (additional Rule 5, see below)
        else:
            features[i]['type'] = 'body'

    for i in range(len(features)):
        if (i == len(features)-1):
            continue
        # Rule 5: Any section whose width is less than 50% of the page width and next section being a footer gets classified
        #         as a 'footer'
        if ((features[i]['width'] < (0.5 * image_sliced.shape[1])) & (features[i+1]['type'] == 'footer')):
            features[i]['type'] = 'footer'

    # Consolidate sections and export to a txt file
    header = ""
    body = ""
    footer = ""
    for item in features:
        if (item['type'] == 'header'):
            header = header + '\n' + item['text']
        if (item['type'] == 'body'):
            body = body + '\n' + item['text']
        if (item['type'] == 'footer'):
            footer = footer + '\n' + item['text']
           
    out_dir_file = image_file.replace(".jpg",".txt")
    with open(out_dir_file, "w") as text_file:
        print("Header: {}\n\nBody: {}\n\nFooter: {}".format(header, body, footer), file=text_file)
    print (image_file.replace(".jpg",".txt") + " created successfully")

    
# Define the main function for standalone script call
if __name__ == '__main__':
    read_letters(sys.argv[1])
    