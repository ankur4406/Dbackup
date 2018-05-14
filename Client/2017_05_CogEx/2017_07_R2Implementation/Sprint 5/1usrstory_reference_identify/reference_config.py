# Setting up by importing all the relevant libraries
import sys
import cv2
import json
import base64
from scipy import misc
import io

def reference_config(img_form, config_file_in):
    """
    
    Purpose: To identify and extract the reference markers for each element of the unknown form based on the 
                co-ordinate based configuration file
    Input(s): 1. Unknown form as a .png file
              2. Coordinate based configuration file (.json), created either manually or using the XPMS UI
    Outputs(s): A re-defined configuration file (.json) replacing the coordinates for each element with a  
                reference marker, encoded as a string (using base64 encoding)
    Created by: Ankur Arora, Deloitte USI
    Created on: July, 31, 2017
    Modified by: 
    Modified on:
    
    """
    
    # Loading the form template in the form of an array
    form = cv2.imread(img_form)
    form_gray = cv2.cvtColor(form,cv2.COLOR_BGR2GRAY)
    form_gray = cv2.GaussianBlur(form_gray,(5,5),0)
    
    # Loading the coordinate based json configuration file as a python dictionary
    with open(config_file_in) as config:
        dict_config = json.load(config)
        
    # Looping through the configuration file, converting coordinates to reference markers
    for key in dict_config.keys():
        x1 = x2 = y1 = y2 = 0.0 # Resetting the cooordinates
        # Defining the X coordinates (x1, x2) of the reference marker
        x1 = dict_config[key]["coordinates"]["x1"]
        x2 = dict_config[key]["coordinates"]["x2"]
        # Defining the Y coordinates (y1, y2) of the reference marker
        if dict_config[key]["type"] == "txt":
            # Slice the region right above the text element
            y1 = dict_config[key]["coordinates"]["y1"] - (dict_config[key]["coordinates"]["y2"] - dict_config[key]["coordinates"]["y1"])
            y2 = dict_config[key]["coordinates"]["y1"]
        if dict_config[key]["type"] == 'omr':
            # Slice the OMR region itself
            y1 = dict_config[key]["coordinates"]["y1"]
            y2 = dict_config[key]["coordinates"]["y2"]
            # Slice the region above the table as the start reference marker
        if dict_config[key]["type"] == 'tbl':
            y1 = dict_config[key]["coordinates"]["y1"] - (100)
            y2 = dict_config[key]["coordinates"]["y1"]
    
        # Encode the marker into a base64 string 
        ref_marker = misc.toimage(form_gray[y1:y2,x1:x2])
        in_mem_file = io.BytesIO()
        ref_marker.save(in_mem_file, format = "PNG")
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        ref_marker_encoded = base64.b64encode(img_bytes)
    
        # Define the Y coordinates (y1, y2) for the end of the table element(s)  
        if dict_config[key]["type"] == 'tbl':
            # Slice the region right after the end of the table
            y1_ = dict_config[key]["coordinates"]["y2"]
            y2_ = dict_config[key]["coordinates"]["y2"] + (100)
            # Encode the marker into a base64 string
            ref_marker = misc.toimage(form_gray[y1_:y2_,x1:x2])
            in_mem_file = io.BytesIO()
            ref_marker.save(in_mem_file, format = "PNG")
            in_mem_file.seek(0)
            img_bytes = in_mem_file.read()
            ref_marker_encoded_ = base64.b64encode(img_bytes)
        else:
            # For non-table elements, no end marker is required
            ref_marker_encoded_ = ""
        
        # Update the dictionary to add the reference markers and remove the coordinates 
        dict_config[key]["ref1"] = ref_marker_encoded
        dict_config[key]["ref2"] = ref_marker_encoded_
        del dict_config[key]["coordinates"]

    # Export the updated dictionary as a new (json) configuration file
    with open(config_file_in.replace(".json","") + "_ref.json", 'w') as fp:
        json.dump(dict_config, fp)
        print (config_file_in.replace(".json","") + "_ref.json created successfully")

# Define the main function for standalone script call
if __name__ == '__main__':
    reference_config(sys.argv[1], sys.argv[2])

