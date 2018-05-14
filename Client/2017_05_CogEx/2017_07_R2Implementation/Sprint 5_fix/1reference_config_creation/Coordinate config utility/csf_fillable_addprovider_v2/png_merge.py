import os
import cv2
import numpy as np

def remove_skewness(image):
    gray = image
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # print("[INFO] angle: {:.3f}".format(angle))
    return(rotated)

def png_merge():
    imgs=[]
    for f in os.listdir(dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        # print(f)
        f1 = cv2.imread(os.path.join(dir, f),0)
        # print(f1)
        imgs.append(remove_skewness(f1))
    cv2.imwrite('merged.png', np.vstack(tuple([img for img in imgs])))

# Define the main function for standalone script call
if __name__ == '__main__':
    dir = os.getcwd()
    valid_images = [".png"]
    png_merge()
    print("Merged png created!!")