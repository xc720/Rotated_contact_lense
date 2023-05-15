import cv2
import numpy as np


import numpy as np
import cv2
import math 
import matplotlib.pyplot as plt

#Calculate the distance from the center to the ellipse 
def ellipse_coordinates(a, b, t, h, k, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    
    x = h + a * np.cos(t) * np.cos(angle_radians) - b * np.sin(t) * np.sin(angle_radians)
    y = k + a * np.cos(t) * np.sin(angle_radians) + b * np.sin(t) * np.cos(angle_radians)

    return x,y

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def Fitted_Ellipse_Analysis(image_path):
    # Read the image
    image = cv2.imread('image_path', cv2.IMREAD_COLOR)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0) #smaller kernels detects fine features

    # Perform edge detection
    edges = cv2.Canny(gray_image, threshold1=50, threshold2=100)

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
    cv2.ellipse(result, ellipse, (0, 255, 0), 2) #!!! get the contour the ellipse represented as a tuple with the following structure: ((h, k), (MA, ma), angle)

    # Calculate the center of mass (centroid) of the ellipse
    center_point = ellipse[0]

    # Draw the center of mass on the result image
    center_x = center_point[0]
    center_y = center_point[1]
    print("Center point of the ellipse:", center_x, center_y)
    print('angle:', ellipse[2])

    #Calculate the distance from the center to the ellipse 
    semi_major_axis_length = (ellipse[0][0])/2   
    semi_minor_axis_length = (ellipse[0][1])/2    


    angle_degrees = ellipse[2]           # Roataion angle of the ellipse


    return (center_x, center_y, semi_major_axis_length, semi_minor_axis_length, angle_degrees)

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def main():
    image_path1 = 'test5\image78-1.tiff'
    image_path2 = 'test5\image78-2.tiff'
    centroid_1 = Fitted_Ellipse_centroid(image_path1)
    centroid_2 = Fitted_Ellipse_centroid(image_path2)
    print(centroid_1)
    
    t = np.linspace(0, 2 * np.pi, 360) # 360 equally spaced points between 0 and 2π



    # Set the maximum distance to search for an edge
    max_distance = 200

    # Sample edges


    # Print the distances
    print(edge_distances_1)
    
if __name__ == '__main__':
    main()