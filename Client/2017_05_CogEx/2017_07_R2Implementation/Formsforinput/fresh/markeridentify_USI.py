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

"""
    Section Added by: Ankur Arora
    Objective: To deskew the input form
"""

gray1 = cv2.bitwise_not(gray)
coords = np.column_stack(np.where(gray1 > 0))
angle = cv2.minAreaRect(coords)[-1]

# the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
if angle < -45:
    angle = -(90 + angle)
# otherwise, just take the inverse of the angle to make it positive
else:
    angle = -angle

# rotate the image to deskew it
(h, w) = gray1.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

"""     End of Section     """


"""
    Section Added by: Ankur Arora
    Objective: To identify the appropriate scaling factor for the form 
                and template based on the closest match
"""

scales = np.arange(0.5, 2.0, 0.1)

scale_large = 0.0
scale_small = 0.0
max_corr = 0.0
for i in scales:
    for j in scales:
        large_image = cv2.resize(rotated,None,fx=i, fy=i, interpolation = cv2.INTER_CUBIC)
        small_image = cv2.resize(template,None,fx=j, fy=j, interpolation = cv2.INTER_CUBIC)

        result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
        _,mx,_,mxLoc = cv2.minMaxLoc(result)
        if (mx > max_corr):
            scale_large = i
            scale_small = j
            max_corr = mx

large_image = cv2.resize(rotated,None,fx=scale_large, fy=scale_large, interpolation = cv2.INTER_CUBIC)
small_image = cv2.resize(template,None,fx=scale_small, fy=scale_small, interpolation = cv2.INTER_CUBIC)

"""     End of Section     """

result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED) # Changed by Ankur Arora
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#Create Bounding Box
top_left = max_loc # Changed by Ankur Arora
print ("Match Score: " + str(max_val)) 
bottom_right = (top_left[0] + small_image.shape[1], top_left[1] + small_image.shape[0]) # Changed by Ankur Arora
cv2.rectangle(large_image, top_left, bottom_right, (0,0,255), 5)

cv2.imwrite(output_image, large_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
