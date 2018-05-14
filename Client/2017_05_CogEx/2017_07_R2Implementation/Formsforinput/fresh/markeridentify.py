import sys
teamplate_path = sys.argv[1]  # 'passporttemplate.png'
image_to_be_processed = sys.argv[2] # 'resized.png'
output_image = sys.argv[3] # 'Markeronresized.png'


import cv2
import numpy as np

# Load input image and convert to grayscale
image = cv2.imread(image_to_be_processed)
#image = cv2.imread('form1.png')
#cv2.imshow('Where is Waldo?', image)
#cv2.waitKey(0)
#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Load Template image
#template = cv2.imread('images/tent.jpg',0)
template = cv2.imread(teamplate_path,0)
#template = cv2.imread('cms_1500_cut_medicare.png',0)
#template = cv2.imread('collegeform_logo.png',0)

result = cv2.matchTemplate(gray, template, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#Create Bounding Box
top_left = min_loc
print(top_left)
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imwrite(output_image, image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
