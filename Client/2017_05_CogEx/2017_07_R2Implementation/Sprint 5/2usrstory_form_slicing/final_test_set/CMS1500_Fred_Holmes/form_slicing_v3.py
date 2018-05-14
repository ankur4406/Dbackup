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
    
    out_dir = os.path.join(os.getcwd(), img_form.replace(".png","") + '_slices')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Looping through the configuration file, converting the reference markers to coordinates
    for key in dict_config.keys():
        # print (key)
        # Extract reference marker strings
        ref1 = dict_config[key]["ref1"]
        ref2 = dict_config[key]["ref2"]
        # Decode marker 1 to image
        ref1_d = base64.decodebytes(ref1.encode('utf-8'))
        with open('temp_image.png', 'wb') as f:
            f.write(ref1_d)
            f.close()
        img_ref1 = cv2.imread('temp_image.png',0)
        os.remove('temp_image.png')
        # Identify the appropriate aspect ratio to use #
        if key == list(dict_config.keys())[0]:
            scales = np.arange(0.5, 2.0, 0.5)
            scale_large = 0.0
            scale_small = 0.0
            max_corr = 0.0
            for i in scales:
                for j in scales:
                    large_image = cv2.resize(form_gray,None,fx=i, fy=i, interpolation = cv2.INTER_CUBIC)
                    small_image = cv2.resize(img_ref1,None,fx=j, fy=j, interpolation = cv2.INTER_CUBIC)
                    if ((small_image.shape[0] >= large_image.shape[0]) | (small_image.shape[1] >= large_image.shape[1])):
                        break
                    result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
                    _,mx,_,mxLoc = cv2.minMaxLoc(result)
                    if (mx > max_corr):
                        scale_large = i
                        scale_small = j
                        max_corr = mx
                    # print (i, j, mx)
            # print (scale_small, scale_large, max_corr)

        # Perform template matching to obtain the best match coordinates & confidence
        large_image = cv2.resize(form_gray,None,fx=scale_large, fy=scale_large, interpolation = cv2.INTER_CUBIC)
        small_image = cv2.resize(img_ref1,None,fx=scale_small, fy=scale_small, interpolation = cv2.INTER_CUBIC)
        # misc.imsave(key + '_ref.png', small_image)
        # print (large_image.shape, small_image.shape)
        result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # Define coordinates for slicing for each element type
        if dict_config[key]["type"] == "txt":
            dict_config[key]["coordinates"] = {"x1": int(max_loc[0]/scale_large),
                                               "x2": int((max_loc[0] + small_image.shape[1])/scale_large),
                                               "y1": int((max_loc[1] + small_image.shape[0])/scale_large),
                                               "y2": int((max_loc[1] + (2 * small_image.shape[0]))/scale_large)}
        if dict_config[key]["type"] == "omr":
            dict_config[key]["coordinates"] = {"x1": int(max_loc[0]/scale_large),
                                               "x2": int((max_loc[0] + small_image.shape[1])/scale_large),
                                               "y1": int(max_loc[1]/scale_large),
                                               "y2": int((max_loc[1] + small_image.shape[0])/scale_large)}
        if dict_config[key]["type"] == "tbl":
            # Extract the 2nd reference marker as well for table elements
            ref2_d = base64.decodebytes(ref2.encode('utf-8'))
            with open('temp_image.png', 'wb') as f:
                f.write(ref2_d)
                f.close()
            img_ref2 = cv2.imread('temp_image.png',0)
            os.remove('temp_image.png')
            # print (large_image.shape, small_image.shape)
            # Perform template match for 2nd marker as well
            small_image = cv2.resize(img_ref2,None,fx=scale_small, fy=scale_small, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
            _, max_val_, _, max_loc_ = cv2.minMaxLoc(result)
            # Define coordinates
            dict_config[key]["coordinates"] = {"x1": int(max_loc[0]/scale_large),
                                               "x2": int((max_loc[0] + small_image.shape[1])/scale_large),
                                               "y1": int((max_loc[1] + 50)/scale_large),
                                               "y2": int(max_loc_[1]/scale_large)}
            
        # Remove references from the dictionary
        del dict_config[key]["ref1"], dict_config[key]["ref2"]
        
        # Section below only for testing
        # print (dict_config[key]["coordinates"]["x1"], dict_config[key]["coordinates"]["y1"], dict_config[key]["coordinates"]["x2"], dict_config[key]["coordinates"]["y2"])
        img_slice = misc.toimage(form_gray[dict_config[key]["coordinates"]["y1"]: dict_config[key]["coordinates"]["y2"],
                                               dict_config[key]["coordinates"]["x1"]: dict_config[key]["coordinates"]["x2"]])
        misc.imsave(os.path.join(out_dir, key + '_slice.png'), img_slice)
        # Testing section end

    # Export the updated dictionary as a new (json) configuration file
    with open(os.path.join(out_dir, config_file_in.replace("_ref","").replace(".json","") + "_coord.json"), 'w') as fp:
        json.dump(dict_config, fp, indent=4)
        print (config_file_in.replace("_ref","").replace(".json","") + "_coord.json created successfully")

# Define the main function for standalone script call
if __name__ == '__main__':
    form_slicing(sys.argv[1], sys.argv[2])