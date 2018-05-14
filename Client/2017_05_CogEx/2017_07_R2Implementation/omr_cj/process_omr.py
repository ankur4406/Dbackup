import cv2, numpy as np
import copy


def crop_get_rois(listOfCBoxes,imshape, XLoffset, XRoffset, YLoffset, YRoffset):
	avgHeight = 0
	avgWidth = 0

	## Calculate avgWidth and avgHeight
	for i in range(len(listOfCBoxes)):
		avgHeight += listOfCBoxes[i][3] - listOfCBoxes[i][1]
		avgWidth += listOfCBoxes[i][2] - listOfCBoxes[i][0]

	avgHeight = 1.0*avgHeight/len(listOfCBoxes)
	avgWidth = 1.0*avgWidth/len(listOfCBoxes)

	avgXLOffset = avgWidth*(XLoffset/100.0)
	avgXROffset = avgWidth*(XRoffset/100.0)
	avgYLOffset = avgHeight*(YLoffset/100.0)
	avgYROffset = avgHeight*(YRoffset/100.0)

	## Add an offset for each check box
	for i in range(len(listOfCBoxes)):
		listOfCBoxes[i][0] = int( max( listOfCBoxes[i][0] + avgXLOffset, 0 ) ) # Can't be negative
		listOfCBoxes[i][1] = int( max( listOfCBoxes[i][1] + avgYLOffset, 0 ) ) # Can't be negative
		listOfCBoxes[i][2] = int( min( listOfCBoxes[i][2] + avgXROffset, imshape[1]-1 ) )
		listOfCBoxes[i][3] = int( min( listOfCBoxes[i][3] + avgYROffset, imshape[0]-1 ) )

	## Calculate ROI
	minx = miny =  10000
	maxx = maxy = -1
	for i in range(len(listOfCBoxes)):
		minx = min( minx, listOfCBoxes[i][0] ) #x1
		miny = min( miny, listOfCBoxes[i][1] ) #y1
		maxx = max( maxx, listOfCBoxes[i][2] ) #x2
		maxy = max( maxy, listOfCBoxes[i][3] ) #y2

	## Update list of CBoxes
	for i in range(len(listOfCBoxes)):
		listOfCBoxes[i][0] = listOfCBoxes[i][0] - minx
		listOfCBoxes[i][1] = listOfCBoxes[i][1] - miny
		listOfCBoxes[i][2] = listOfCBoxes[i][2] - minx
		listOfCBoxes[i][3] = listOfCBoxes[i][3] - miny

	return int(minx),int(miny),int(maxx),int(maxy), listOfCBoxes

def get_field_sum_list(im, listOfCBoxes, XLoffset, YLoffset, XRoffset,YRoffset):
	x1,y1,x2,y2, listOfCBoxes_ = crop_get_rois(listOfCBoxes,im.shape,  XLoffset, XRoffset, YLoffset, YRoffset)
	field_im = im[y1:y2,x1:x2]
	
	field_im = cv2.fastNlMeansDenoising(cv2.blur(field_im, (3, 3)), 30, 30, 15, 50)
	_, field_im = cv2.threshold(~field_im, 127, 255, 0)
#	field_im = ~cv2.morphologyEx(field_im, cv2.MORPH_CLOSE, 
#									np.ones((3, 3), np.uint8), iterations=3)

	field_im_c = cv2.cvtColor(field_im, cv2.COLOR_GRAY2BGR)

	field_sum_list = []
	for i in range(len(listOfCBoxes_)):
		x1 = listOfCBoxes_[i][0]
		y1 = listOfCBoxes_[i][1]
		x2 = listOfCBoxes_[i][2]
		y2 = listOfCBoxes_[i][3]
		area = (x2-x1)*(y2-y1)
		fsum = np.sum( field_im[y1:y2, x1:x2] )/(255.0)
		field_sum_list.append( fsum )
		
		field_im_c = cv2.rectangle(field_im_c, (x1,y1),(x2,y2), (0,0,255))

	# cv2.imshow("t",field_im_c )
	# cv2.waitKey(0)

	field_sum_list = np.array(field_sum_list)
	ssum = np.sum(field_sum_list)
	if( len(listOfCBoxes_) == 1 ): # Single_box
		field_sum_list = np.array(field_sum_list)/area
	else:	
		if( ssum > 0 ):
			field_sum_list = np.array(field_sum_list)/ssum
	
	return field_sum_list

def get_multi_indices_median(A): ## any value greater than medium, return as marked
	indices = []
	medianV = np.median(A)
	
	if( medianV < 0.0001 ):
		divideBy = 1
	else:
		divideBy = medianV

	for i in range(len(A)):
		v = 100.0*(A[i]-medianV)/divideBy
		if( v > 1 ):
			indices.append(i)
	return indices

def get_multi_indices_greater_than_min(A):
	if( len(A) == 0 ):
		return []
	return np.where(A > np.min(A))[0]

def get_check_boxes_confidence(im, listOfCBoxes, isMulti=False):
	if( len(im.shape) == 3 ): #If color make it gray scale
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# A = get_field_sum_list(im, copy.deepcopy(listOfCBoxes), 20,20,-20,-20) 
	# A +=  get_field_sum_list(im, copy.deepcopy(listOfCBoxes), -20,0,0,0) 
	# A +=  get_field_sum_list(im, copy.deepcopy(listOfCBoxes), 0,0,20,0) 
	# A +=  get_field_sum_list(im, copy.deepcopy(listOfCBoxes), 0,-20,0,0) 
	# A +=  get_field_sum_list(im, copy.deepcopy(listOfCBoxes), 0,0,0,20) 
	indices = []
	if( len(listOfCBoxes) == 1 ):
		A = get_field_sum_list(im, copy.deepcopy(listOfCBoxes), 20,20,-20,-20) 
		if(A[0]>0.01):
			indices.append(0)
		return A,indices
	
	if( not isMulti ):
		A = get_field_sum_list(im, copy.deepcopy(listOfCBoxes), -20,20,20,-20) 
		indices.append( np.argmax(A)  )
	else:	
		A = get_field_sum_list(im, copy.deepcopy(listOfCBoxes), 20,20,-20,-20) 		
		indices = get_multi_indices_greater_than_min(A)
		if( (len(indices) == len(A)) or (len(indices) == 0) ):
			indices = []
			A = get_field_sum_list(im, copy.deepcopy(listOfCBoxes), -20,20,20,-20) 
			indices = get_multi_indices_greater_than_min(A)
			
	return A,indices



img_fname = "omr1.jpg"
configfname = ''.join( img_fname.split(".")[:-1] )+".config"
listOfCBoxes = []
for line in open(configfname,"r"):
	listOfCBoxes.append( list( map( int, line.split(",") ) ) )


print( get_check_boxes_confidence(cv2.imread(img_fname), listOfCBoxes, isMulti=True) )

cv2.imshow( "tm", cv2.imread(img_fname) )
cv2.waitKey(0)
