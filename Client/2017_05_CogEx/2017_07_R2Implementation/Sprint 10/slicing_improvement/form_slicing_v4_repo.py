"""

Purpose: To match the reference markers for each element of the known form type based on the reference 
         based configuration file and slice the form
Input(s): 1. Instance of a known form type as a .png file
          2. Reference based configuration file (.json), created using reference_config.py
Outputs(s): 1. A re-defined configuration file (.json) replacing the reference markers for each element with a  
            coordinate map particular to the form instance
            2. Individual slices stored as separate png files
Created by: Ankur Arora, Deloitte USI
Created on: August, 29, 2017
Modified by: 

"""

# Setting up by importing all the relevant libraries
import os
import sys
import numpy as np
import cv2
from scipy import misc
import json
import base64
from concurrent.futures.thread import ThreadPoolExecutor

def form_slicing(img_form, dict_config ,out_dir):
    # Loading the form instance as an array
    form = cv2.imread(img_form)
    form_gray = cv2.cvtColor(form,cv2.COLOR_BGR2GRAY)
    form_gray = cv2.GaussianBlur(form_gray,(5,5),0)
    form_gray = form_gray.astype(np.uint8)

    # Loading the reference based json configuration file as a python dictionary
    with open(dict_config) as config:
        dict_config = json.load(config)

    # Create output directory

    # Identify the appropriate aspect ratio to use
    print ("\nIdentifying the appropriate aspect ratio to use...")
    key = "Form_Header"
    ref1 = dict_config[key]["ref1_img"]
    # Decode marker to image

    ref1_d = base64.decodebytes(ref1.encode('utf-8'))
    with open('temp_image.png', 'wb') as f:
        f.write(ref1_d)
        f.close()
    img_ref1 = cv2.imread('temp_image.png',0)
    img_ref1 = img_ref1.astype(np.uint8)
    os.remove('temp_image.png')

    scales = np.arange(0.8, 1.2, 0.2)
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
    print ("\n Aspect ratio summary \n Ingested form scaled to: {}x \n Reference markers scaled to: {}x \n".format(
        scale_large, scale_small))
    del dict_config[key]
    print("*"*100)
    dict_config = repeat_sections(dict_config) # Change 2
        
    # Looping through the configuration file, converting the reference markers to coordinates
    def do_processing(key):
        print ("Extracting form slice for field {}...".format(key))
        temp_image_name = key + '_temp_image.png'
        # Extract reference marker strings
        ref1 = dict_config[key]["ref1_img"]
        # Decode marker to image
        ref1_d = base64.decodebytes(ref1.encode('utf-8'))
        with open(temp_image_name, 'wb') as f:
            f.write(ref1_d)
            f.close()
        img_ref1 = cv2.imread(temp_image_name,0)
        img_ref1 = img_ref1.astype(np.uint8)
        os.remove(temp_image_name)

        # Perform template matching to obtain the best match coordinates & confidence
        large_image = cv2.resize(form_gray,None,fx=scale_large, fy=scale_large, interpolation = cv2.INTER_CUBIC)
        small_image = cv2.resize(img_ref1,None,fx=scale_small, fy=scale_small, interpolation = cv2.INTER_CUBIC)
        result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if dict_config[key]['type']=='tbl':
            ref2 = dict_config[key]["ref2_img"]
            # Decode marker to image
            ref2_d = base64.decodebytes(ref2.encode('utf-8'))
            with open(temp_image_name, 'wb') as f:
                f.write(ref2_d)
                f.close()
            img_ref2 = cv2.imread(temp_image_name,0)
            img_ref2 = img_ref2.astype(np.uint8)
            os.remove(temp_image_name)

            small_image_ = cv2.resize(img_ref2,None,fx=scale_small, fy=scale_small, interpolation = cv2.INTER_CUBIC)
            result_ = cv2.matchTemplate(large_image, small_image_, cv2.TM_CCOEFF_NORMED)
            _, max_val_, _, max_loc_ = cv2.minMaxLoc(result_)
            
            del dict_config[key]["ref2_img"]


        # Define coordinates for slicing for each element type
        # print (max_loc)
        drift = dict_config[key]['drift']
        x1 = int(max_loc[0] / scale_large) + int(drift['x1'] / scale_large)
        x2 = int((max_loc[0] + small_image.shape[1]) / scale_large) + int(drift['x2'] / scale_large)
        y1 = int(max_loc[1] / scale_large) + int(drift['y1'] / scale_large)
        if dict_config[key]['type'] == 'tbl':
            # print(max_loc_)
            y2 = int((max_loc_[1] + small_image_.shape[0]) / scale_large) + int(drift['y2'] / scale_large)
            confidence = np.mean((max_val, max_val_))
        else:
            y2 = int((max_loc[1] + small_image.shape[0]) / scale_large) + int(drift['y2'] / scale_large)
            confidence = max_val

        slc = misc.toimage(form_gray[y1:y2, x1:x2]) # Change 1

        # Export the slice and update the config file 
        misc.imsave(os.path.join(out_dir, key + '_slice.png'), slc)
        del dict_config[key]["ref1_img"], dict_config[key]["drift"]
        dict_config[key]["slice_file"] = key + '_slice.png'
        dict_config[key]["slice_xy"] = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
        dict_config[key]["confidence"] = confidence

    dict_config_keys = dict_config.keys()
    with ThreadPoolExecutor(max_workers=2) as executor:
        for key in dict_config_keys:
            executor.submit(do_processing, key)
    return dict_config

def clean_flow(flow_file):
    for item in flow_file['resources']:
        if 'classification' in item:
            del item['classification']['configuration']
    return flow_file

def repeat_sections(config):
    for key in config.keys():
        if config[key]['type'] == 'sec':
            print(config[key])
    return config

if __name__ == "__main__":
    img_form = r'C:\Users\ankuarora\Desktop\Client\2017_05_CogEx\2017_07_R2Implementation\Sprint 10\slicing_improvement\sample_forms\preprocessed_xpms.png' 
    dict_config = r'Provider_config_base64_improve.json'
    out_dir = r'C:\Users\ankuarora\Desktop\Client\2017_05_CogEx\2017_07_R2Implementation\Sprint 10\slicing_improvement\slices'
    form_slicing(img_form, dict_config ,out_dir)
