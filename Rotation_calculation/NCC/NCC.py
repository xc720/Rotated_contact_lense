import cv2
import numpy as np


def Fitted_Ellipse_centroid(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) #eg.test5\image78-1.tiff
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #smaller kernels detects fine features

    # Perform edge detection
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=100)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fit an ellipse to the contour with the largest area
    max_area = 0
    ellipse = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ellipse = cv2.fitEllipse(cnt)

    # Draw the ellipse
    result = image.copy()
    cv2.ellipse(result, ellipse, (0, 255, 0), 2)

    # Calculate the center of mass (centroid) of the ellipse
    center_point = ellipse[0]


    # Draw the center of mass on the result image
    center_x = int(round(center_point[0]))
    center_y = int(round(center_point[1]))
    # print("Center point of the ellipse:", center_x, center_y)

    cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)

    return (center_x, center_y)

def main():
    image_path1 = 'test5\image78-1.tiff'
    image_path2 = 'test5\image78-2.tiff'
    centroid_1 = Fitted_Ellipse_centroid(image_path1)
    centroid_2 = Fitted_Ellipse_centroid(image_path2)
    print(centroid_1)
    
    
if __name__ == '__main__':
    main()