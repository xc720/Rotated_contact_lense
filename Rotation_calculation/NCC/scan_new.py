import cv2
import numpy as np    
import math 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import correlate


#Calculate the distance from the center to the ellipse 
def ellipse_coordinates(a, b, t, h, k, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    
    x = h + a * np.cos(t) * np.cos(angle_radians) - b * np.sin(t) * np.sin(angle_radians)
    y = k + a * np.cos(t) * np.sin(angle_radians) + b * np.sin(t) * np.cos(angle_radians)

    return x,y

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def Fitted_Ellipse_Distance_Analysis(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
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
    # print('angle:', ellipse[2])

    #Calculate the distance from the center to the ellipse 
    # print("semi(MA,ma):",ellipse[1])
    
    
    semi_major_axis_length = max(ellipse[1][0],ellipse[1][1]) /2   
    semi_minor_axis_length = min(ellipse[1][0],ellipse[1][1]) /2    
    print('semi_major_axis_length', semi_major_axis_length)
    print('semi_minor_axis_length', semi_minor_axis_length)
    t = np.linspace(0, 2 * np.pi, 360) # 360 equally spaced points between 0 and 2π

    angle_degrees = ellipse[2]           # Roataion angle of the ellipse

    x, y = ellipse_coordinates(semi_major_axis_length , semi_minor_axis_length , t, center_x, center_y, angle_degrees)
    
    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Draw the original ellipse
    # ellipse_outline = patches.Ellipse(center_point, ellipse[1][0], ellipse[1][1], 
    #                                 angle=ellipse[2], fill=False, edgecolor='green')
    # ax.add_patch(ellipse_outline)

    # Create a scatter plot of the (x, y) points
    plt.scatter(x, y, label='Points on Ellipse', color='red', marker='o')

    # Optionally, you can also plot the center of the ellipse
    plt.scatter(center_x, center_y, label='Center', color='blue', marker='x')

    # Add labels, title, and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Ellipse Points')
    plt.legend()

    # Display the plot
    plt.show()

    # Calculate distances
    distances = [euclidean_distance(x_i, y_i, center_x, center_y) for x_i, y_i in zip(x, y)]
    

    # Print distances
    # for i, distance in enumerate(distances):
    #     print(f"Distance between point {i+1} ({x[i]}, {y[i]}) and center ({center_x}, {center_y}): {distance}")
    print('angle of ellipse:',angle_degrees)       
    
    cv2.imwrite(f'Detected_Circle_Edge{angle_degrees}.jpg', result)
    print('saved')
    return distances,angle_degrees 
        


    # Display the result

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

def sign_change(distance_list1):
    y1 = np.array(distance_list1)

    # Compute differences between consecutive y-values
    dy = np.diff(y1)

    # Compute the sign of the differences
    signs = np.sign(dy)

    # Find where the sign changes
    sign_changes1 = np.where(np.diff(signs) != 0)[0] + 1

    print("Indices where the derivative changes sign:", sign_changes1)
    return sign_changes1    
    
    
def main():
    image_path1 = 'test5\image7-1.tiff'
    image_path2 = 'test5\image7-2.tiff'
    distance_list1 , angle_degree1 = Fitted_Ellipse_Distance_Analysis(image_path1)
    distance_list2 , angle_degree2 = Fitted_Ellipse_Distance_Analysis(image_path2)
    
    print(angle_degree1)
    print(angle_degree2)
    print('rough roatation angle:', abs(angle_degree2-angle_degree1))
    
    # correlations = np.correlate(distance_list1, distance_list2, "full")
    # estimated_shift = correlations.argmax() - (len(distance_list1) - 1)

    # print(f"Estimated shift: {estimated_shift}")
    
    # Create a line plot
    x1 = np.arange(len(distance_list1))
    plt.plot(x1, distance_list1, label='distance_list1')
    x2 = np.arange(len(distance_list2))
    plt.plot(x2, distance_list2, label='distance_list2')
    plt.legend()
    
    distance_list1_np = np.array(distance_list1)
    distance_list2_np = np.array(distance_list2)
    
    # Cross-correlation of the two signals
    cross_correlation = correlate(distance_list1_np, distance_list2_np)

    # The phase difference is the argument that maximizes the cross-correlation
    phase_difference = np.argmax(cross_correlation) - (len(distance_list1_np) - 1)

    print("Phase difference:", phase_difference)





    # Display the plot
    plt.show()
    


if __name__ == '__main__':
    main()