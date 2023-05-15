import numpy as np
import cv2
 
# Read image
img = cv2.imread('test5\image84-1.tiff')
# Smooth it
img = cv2.medianBlur(img,3)
img_copy = img.copy()
# Convert to greyscale
img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)
# Apply Hough transform to greyscale image
circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,20,
                     param1=60,param2=40,minRadius=650,maxRadius=660)
circles = np.uint16(np.around(circles))
# Draw the circles
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow('detected circles',img)
# Save the cropped image
cv2.imwrite( 'detected_circles.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()