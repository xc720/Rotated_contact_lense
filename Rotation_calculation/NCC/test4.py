import numpy as np
import math

def ellipse_coordinates(a, b, t, h, k, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    
    x = h + a * np.cos(t) * np.cos(angle_radians) - b * np.sin(t) * np.sin(angle_radians)
    y = k + a * np.cos(t) * np.sin(angle_radians) + b * np.sin(t) * np.cos(angle_radians)
    
    return x, y

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Example usage
semi_major_axis_length = 10  # Replace with your value
semi_minor_axis_length = 5   # Replace with your value
t = np.linspace(0, 2 * np.pi, 100) # 100 equally spaced points between 0 and 2π
center_x = 0                 # Replace with your value
center_y = 0                 # Replace with your value
angle_degrees = 30           # Replace with your value

x, y = ellipse_coordinates(semi_major_axis_length , semi_minor_axis_length , t, center_x, center_y, angle_degrees)

# Calculate distances
distances = [euclidean_distance(x_i, y_i, center_x, center_y) for x_i, y_i in zip(x, y)]

# Print distances
for i, distance in enumerate(distances):
    print(f"Distance between point {i+1} ({x[i]}, {y[i]}) and center ({center_x}, {center_y}): {distance}")
