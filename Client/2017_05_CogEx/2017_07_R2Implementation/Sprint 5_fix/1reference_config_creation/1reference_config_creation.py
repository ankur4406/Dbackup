"""

Purpose: To create a reference based configuration file by encoding the reference marker images into base64 encoding
Input(s): 1. Unknown form as a .png file
          2. Coordinate based configuration file (.json), created either manually or using the XPMS UI
Outputs(s): A re-defined configuration file (.json) replacing the coordinates for each reference marker, encoded as 
            a string (using base64 encoding)
Created by: Ankur Arora, Deloitte USI
Created on: August, 29, 2017
Modified by: 
Modified on:

"""

# Setting up by importing all the relevant libraries
import sys
import cv2
import json
import base64
from scipy import misc
import io
import numpy as np

def reference_config_create(img_form, config_file_in):
    
    # Loading the form template in the form of an array
    form = cv2.imread(img_form)
    form_gray = cv2.cvtColor(form,cv2.COLOR_BGR2GRAY)
    form_gray = cv2.GaussianBlur(form_gray,(5,5),0)
    form_gray = form_gray.astype(np.uint8)
    # form_gray = cv2.threshold(form_gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Deskew the form to get perfect markers
    flip = cv2.bitwise_not(form_gray)
    thresh = cv2.threshold(flip, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    # print(angle)
    (h, w) = form_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(form_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Loading the coordinate based json configuration file as a python dictionary
    with open(config_file_in) as config:
        dict_config = json.load(config)
    print(len(dict_config.keys()))        
    # Looping through the configuration file, converting coordinates to images
    for key in dict_config.keys():
        # Identify coordinates of the marker
        ref1 = dict_config[key]["ref1_xy"]
        x1 = ref1["x1"]
        x2 = ref1["x2"]
        y1 = ref1["y1"]
        y2 = ref1["y2"]
        
        # Slice the marker from the input form provided
        ref1_slc = misc.toimage(form_gray[y1:y2, x1:x2])

        # Encode the marker into a base64 string 
        in_mem_file = io.BytesIO()
        ref1_slc.save(in_mem_file, format = "PNG")
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        ref1_encoded = base64.encodebytes(img_bytes).decode('utf-8')
        
        # Update dictionary to add reference encodings and removing coordinates
        dict_config[key]["ref1_img"] = ref1_encoded
        del dict_config[key]["ref1_xy"]
        
        # Handle the additional ref marker for tables
        if dict_config[key]["type"] == 'tbl':
            # Identify coordinates of the marker
            ref2 = dict_config[key]["ref2_xy"]
            x1 = ref2["x1"]
            x2 = ref2["x2"]
            y1 = ref2["y1"]
            y2 = ref2["y2"]

            # Slice the marker from the input form provided
            ref2_slc = misc.toimage(form_gray[y1:y2, x1:x2])

            # Encode the marker into a base64 string 
            in_mem_file = io.BytesIO()
            ref2_slc.save(in_mem_file, format = "PNG")
            in_mem_file.seek(0)
            img_bytes = in_mem_file.read()
            ref2_encoded = base64.encodebytes(img_bytes).decode('utf-8')

            # Update dictionary to add reference encodings and removing coordinates
            dict_config[key]["ref2_img"] = ref2_encoded
            del dict_config[key]["ref2_xy"]

    # Export the updated dictionary as a new (json) configuration file
    with open(config_file_in.replace(".json","") + "_base64.json", 'w') as fp:
        json.dump(dict_config, fp, indent=4)
        print (config_file_in + " successfully encoded as " + config_file_in.replace(".json","") + "_base64.json")

# Define the main function for standalone script call
if __name__ == '__main__':
    reference_config_create(sys.argv[1], sys.argv[2])