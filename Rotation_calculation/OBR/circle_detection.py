import cv2
import os
import numpy as np

# Read the image and convert to grayscale
image_path = "test5\image84-1.tiff"
image_name = os.path.basename(image_path)
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Apply Hough Circle Transform to detect circles
circles = circles = cv2.HoughCircles(blurred_image,cv2.HOUGH_GRADIENT,1,20,
                     param1=60,param2=40,minRadius=650,maxRadius=660)
circles = np.uint16(np.around(circles))

# Find the circle with the highest accumulator value (param2)
max_acc_value = 0
selected_circle = None
for circle in circles[0, :]:
    x, y, r = circle
    if r > max_acc_value:
        max_acc_value = r
        selected_circle = circle
        


# Get the bounding box of the circle
x, y, r = selected_circle
x, y, r = int(x), int(y), int(r)
x1, y1, x2, y2 = x - r, y - r, x + r, y + r

# Crop the image using the bounding box
cropped_image = image[y1:y2, x1:x2]

# Save the cropped image
cv2.imwrite(image_name, cropped_image)

# Display the original and cropped images (optional)
cv2.imshow("Original Image", image)
cv2.imshow("Cropped Image", selected_circle)
# cv2.imwrite( 'Cropped_circles.jpg', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
