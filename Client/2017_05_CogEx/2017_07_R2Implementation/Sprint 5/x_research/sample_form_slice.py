# Setting up by importing all the relevant libraries
import os
import cv2
from scipy import misc
import json
import sys

def sample_module(form, config, el_type): # el_type values: txt/tbl/omr
    # Loading the form instance as an array
    form = cv2.imread(form)
    
    # Loading the json configuration file as a python dictionary
    with open(config) as data_file:
        dict_config = json.load(data_file)

    for key in dict_config.keys():
        if dict_config[key]['type'] == el_type:
            img_slice = form[dict_config[key]["coordinates"]["y1"]: dict_config[key]["coordinates"]["y2"], 
                             dict_config[key]["coordinates"]["x1"]: dict_config[key]["coordinates"]["x2"]]
            '''
                Processing code goes here!!
            '''

            
# Define the main function for standalone script call
if __name__ == '__main__':
    sample_module(sys.argv[1], sys.argv[2], sys.argv[3])