import os
import cv2
from scipy import misc
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np

dir = r'C:\Users\ankuarora\Desktop\Client\2017_05_CogEx\2017_07_R2Implementation\Sprint 7\1CreateTrainData\SampleForms'

# Import all images (jpg, gif & png) in the directory as a list
imgs = []
valid_images = [".jpg",".jpeg",".gif",".png"]
for f in os.listdir(dir):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append((f, cv2.imread(os.path.join(dir,f),0)))

# List of initial features that needs be extracted
feature_list = ['width','height','seq_from_top','seq_from_bottom','width_prv',
                'height_prv','width_nxt','height_nxt','text']

# Loop through all images and slice the documents
# Also, create the initial feature space
features = pd.DataFrame(index=[], columns=feature_list)
features.index.name = "filename"
cnt = 0
for img in imgs:
    cnt+=1
    print (str(cnt) + ". Parsing '" + img[0] + "' ...")
    
    # Preprocess the image
    image = cv2.threshold(img[1], 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Slice the document to remove the extra whitespace borders, if any
    p_image = cv2.bitwise_not(image)
    p_image = cv2.threshold(p_image, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(p_image > 0))
    x,y,w,h = cv2.boundingRect(coords)
    image_sliced = image[np.max((x-10,0)):np.min((x+w+10,image.shape[0])),
                         np.max((y-10,0)):np.min((y+h+10, image.shape[1]))]
    
    # Find lines by horizontally blurring the image and thresholding
    blur = cv2.blur(image_sliced, (91,9))
    b_mean = np.mean(blur, axis=1)/255
    threshold = np.percentile(b_mean, 50)
    t = b_mean > threshold
    byte_lines = np.where(1-t)
    byte_lines = byte_lines[0]
    
    # Calculate the median linespace value for defining sections 
    linspace = []
    for x in range(byte_lines.shape[0]-1):
        if byte_lines[x+1] == byte_lines[x] + 1:
            continue
        linspace.append(byte_lines[x+1]-byte_lines[x]-1)
    linspace_limit = (1.0 * np.median(linspace))

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
    # Also, create basic features for use in learning.
    for i in range(len(txt_lines_x)):
        slc = image_sliced[np.max((txt_lines_y[i][0] - 2,0)) : np.min((txt_lines_y[i][1] + 2,image.shape[0])), 
                           np.max((txt_lines_x[i][0] - 2,0)) : np.min((txt_lines_x[i][1] + 2,image.shape[1]))]
        filename = os.path.splitext(img[0])[0] + '_slice' + str(i+1) + os.path.splitext(img[0])[1]
        misc.imsave(os.path.join(dir, "slices\\" + filename), slc)
        ocr_txt = pytesseract.image_to_string(Image.open(os.path.join(dir, "slices\\" + filename)))
        features.loc[filename] = [(txt_lines_x[i][1] - txt_lines_x[i][0] + 4)/image_sliced.shape[1],
                                  (txt_lines_y[i][1] - txt_lines_y[i][0] + 4)/image_sliced.shape[0],
                                  (i+1),
                                  (len(txt_lines_y)-i),
                                  0 if not i else (txt_lines_x[i-1][1] - txt_lines_x[i-1][0] + 4)/image_sliced.shape[1],
                                  0 if not i else (txt_lines_y[i-1][1] - txt_lines_y[i-1][0] + 4)/image_sliced.shape[0],
                                  0 if i==len(txt_lines_x)-1 else (txt_lines_x[i+1][1] - txt_lines_x[i+1][0] + 4)/image_sliced.shape[1],
                                  0 if i==len(txt_lines_y)-1 else (txt_lines_y[i+1][1] - txt_lines_y[i+1][0] + 4)/image_sliced.shape[0],
                                  ocr_txt.lower().replace('\n',' ')
                                 ]

# Dump the features list as csv for manual annotation
features['type'] = ""
features.to_csv('features_raw.csv', encoding='utf-8')

print ("Features csv created successfully!!")