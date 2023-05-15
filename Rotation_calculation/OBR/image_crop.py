import cv2
import os


directory_path = 'test5'
for file_name in os.listdir(directory_path):
            
    file_path = os.path.join(directory_path, file_name)
    # Read the image and convert to grayscale
    image_path = file_path
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 100, 200)

    # Find contours in the binary image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the contour with the largest area
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Save the cropped image
    cv2.imwrite( file_name, cropped_image)



