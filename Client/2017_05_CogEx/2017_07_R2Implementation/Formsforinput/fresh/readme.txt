# The bigger problem we are trying to solve is to process same type of images for text extraction on images with different aspect ratios, tilt or with different sizes.
# One approach we followed for the images having border, and they are working perfectly fine. (Using Opencv border identification feature)
# The problem is with the images without border, so we are using another approach. The pseudo code for that approach is as follow:

   ## Identify marker1 and marker2 (*****HAVING PROBLEM IN THIS STEP ONLY ******)
   ## Compare two markers position and find the aspect ratio, resize , tilt etc. and do the calcualtion as for the standard template

So the problem is if we resize image not able to identify marker. We tried OpenCV feature extraction (SIFT | SURF | ORB | FAST ), none seems work. Than we tried template matching feature of opencv, which is working to some extent. The problem is when we resize the image the marker identification through template matching does not work.

Please find the simplified script "markeridentify.py" , to run the script 

python markeridentify.py  <image_template_to_be_identified>   <name_of_input_image>  <name_of_output_image>

**** Test Case 1 (Standard Working Fine)
E.g. We ran 

python3 markeridentify.py   marker_template_to_be_identified.png  image_processing1.jpg output_image1.png

And output generated is output_image1.png, which seems correctly identified the marker.


**** Test Case 2 (Tilted not resized is also Working Fine)
E.g. We ran 

python3 markeridentify.py   marker_template_to_be_identified.png  image_processing2.jpg output_image2.png

And output generated is output_image2.png, which seems correctly identified the marker.

**** Test Case 3 (If resized it's not working fine)
E.g. We ran 

python3 markeridentify.py   marker_template_to_be_identified.png  image_processing3.jpg output_image3.png

And output generated is output_image3.png, which incorrectly identified the marker.


**** Test Case 4 (If resized and tilted it's not working )

python3 markeridentify.py   marker_template_to_be_identified.png  image_processing4.jpg output_image4.png

And output generated is output_image4.png, which incorrectly identified the marker.


**** Test Case 3 (Not working fine)


We are attaching some sample images.
E.g. we have marker_template_to_be_identified.png
 


