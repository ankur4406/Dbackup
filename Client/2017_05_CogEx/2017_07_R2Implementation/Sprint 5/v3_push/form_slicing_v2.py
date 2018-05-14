"""

Purpose: To match the the reference markers for each element of the known form type based on the reference 
         based configuration file and slice the form
Input(s): 1. Instance of a known form type as a .png file
          2. Reference based configuration file (.json), created using reference_config.py
Outputs(s): A re-defined configuration file (.json) replacing the reference markers for each element with a  
            coordinate map particular to the form instance
Created by: Ankur Arora, Deloitte USI
Created on: August, 01, 2017
Modified by: 
Modified on: 

"""

# Setting up by importing all the relevant libraries
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import json
from pprint import pprint
import io
import base64

def form_slicing(img_form, config_file_in):
    
    # Loading the form instance as an array
    form = cv2.imread(img_form)
    form_gray = cv2.cvtColor(form,cv2.COLOR_BGR2GRAY)
    form_gray = cv2.GaussianBlur(form_gray,(5,5),0)

    # Loading the reference based json configuration file as a python dictionary
    with open(config_file_in) as data_file:
        dict_config = json.load(data_file)

    # Looping through the configuration file, converting the reference markers to coordinates
    for key in dict_config.keys(): 
        # Extract reference marker strings
        ref1 = dict_config[key]["ref1"]
        ref2 = dict_config[key]["ref2"]
        # Decode marker 1 to image
        ref1_d = base64.decodebytes(ref1.encode('utf-8'))
        with open('temp_image.png', 'wb') as f:
            f.write(ref1_d)
            f.close()
        img_ref1 = cv2.imread('temp_image.png',0)
        # Perform template matching to obtain the best match coordinates & confidence
        large_image = cv2.resize(form_gray,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
        small_image = cv2.resize(img_ref1,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
        result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Define coordinates for slicing for each element type
        if dict_config[key]["type"] == "txt":
            dict_config[key]["coordinates"] = {"x1": int(max_loc[0]),
                                               "x2": int(max_loc[0] + small_image.shape[1]),
                                               "y1": int(max_loc[1] + small_image.shape[0]),
                                               "y2": int(max_loc[1] + (2 * small_image.shape[0]))}
        if dict_config[key]["type"] == "omr":
            dict_config[key]["coordinates"] = {"x1": int(max_loc[0]),
                                               "x2": int(max_loc[0] + small_image.shape[1]),
                                               "y1": int(max_loc[1]),
                                               "y2": int(max_loc[1] + small_image.shape[0])}
        if dict_config[key]["type"] == "tbl":
            # Extract the 2nd reference marker as well for table elements
            ref2_d = base64.decodebytes(ref2.encode('utf-8'))
            with open('temp_image.png', 'wb') as f:
                f.write(ref2_d)
                f.close()
            img_ref2 = cv2.imread('temp_image.png',0)
            # Perform template match for 2nd marker as well
            small_image = cv2.resize(img_ref2,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
            _, max_val_, _, max_loc_ = cv2.minMaxLoc(result)
            # Define coordinates
            dict_config[key]["coordinates"] = {"x1": int(max_loc[0]),
                                               "x2": int(max_loc[0] + small_image.shape[1]),
                                               "y1": int(max_loc[1] + 100),
                                               "y2": int(max_loc_[1])}
            
        # Remove references from the dictionary
        del dict_config[key]["ref1"], dict_config[key]["ref2"]
        
        # Section below only for testing
        # img_slice = misc.toimage(large_image[dict_config[key]["coordinates"]["y1"]: dict_config[key]["coordinates"]["y2"],
        #                                       dict_config[key]["coordinates"]["x1"]: dict_config[key]["coordinates"]["x2"]])
        # misc.imsave(key + '_slice.png', img_slice)
        # Testing section end

    # Export the updated dictionary as a new (json) configuration file
    with open(config_file_in.replace("_ref","").replace(".json","") + "_coord.json", 'w') as fp:
        json.dump(dict_config, fp, indent=4)
        print (config_file_in.replace("_ref","").replace(".json","") + "_coord.json created successfully")

# Define the main function for standalone script call
if __name__ == '__main__':
    form_slicing(sys.argv[1], sys.argv[2])