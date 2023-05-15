import cv2
import os
import time
import re
import numpy as np

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def detect_and_compute_features(gray_image):
    orb = cv2.ORB_create() # creates an instance of the ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor
    keypoints, descriptors = orb.detectAndCompute(gray_image, None) 
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #Brute-Force (BF) matcher, cv2.NORM_HAMMING is used, which is suitable for binary descriptors
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance) #sorts the list of matches based on their distance, which represents the similarity between the matched descriptors
    return matches

def resize_image(image, new_width):
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def get_ground_truth_output_name(input_file_name):
    # Extract the number from the input file name
    number = int(re.search(r'(\d+)', input_file_name).group(1))
    
    # Determine the output number based on the input number
    if input_file_name.endswith('-1.tiff'):
        output_number = number 
        output_suffix = '-2.tiff'
    elif input_file_name.endswith('-2.tiff'):
        output_number = number 
        output_suffix = '-1.tiff'
    else:
        raise ValueError("Input file name format not supported")
    
    # Generate the ground truth output name
    output_file_name = f"image{output_number}{output_suffix}"
    return output_file_name

def crop_out(image_path):
    # Read the image and convert to grayscale

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
    
    return cropped_image


def main():
        counter = 0
        # Load, preprocess, and compute features for the input image
        start_time = time.time() 
        print('testing')
        # Set the path to the directory containing the set of files
        directory_path = 'test5'

        input_image_path = 'test5\image95-2.tiff'

        input_image_name = os.path.basename(input_image_path)
        input_image = load_and_preprocess(input_image_path)
        input_keypoints, input_descriptors = detect_and_compute_features(input_image)
        
        


        # Initialize variables to store the best match information
        best_match_file = None
        best_match_num_matches = -1
        best_match_avg_distance = np.inf

        # Iterate through the files in the directory
        for file_name in os.listdir(directory_path):
            
            file_path = os.path.join(directory_path, file_name)
            # print(f"Compare: {input_image_path} with {file_path} ")
            
            # Skip the input image itself
            if file_path == input_image_path:
                print('skip input itself')
                continue

            # Load, preprocess, and compute features for the current file
            current_image = load_and_preprocess(file_path)
            current_keypoints, current_descriptors = detect_and_compute_features(current_image)

            # Match the features of the input image with the features of the current file
            matches = match_features(input_descriptors, current_descriptors)
            

            # Calculate the average distance of the matches
            avg_distance = sum(match.distance for match in matches) / len(matches) 
            
            # Evaluate the quality of the matches (e.g., using the number of matches or the average distance between matches)
            num_matches = len(matches)
            avg_distance = np.mean([match.distance for match in matches])       

            # Update the best match information if the current file has a better match
            if num_matches > best_match_num_matches or (num_matches == best_match_num_matches and avg_distance < best_match_avg_distance):
                best_match_file = file_name
                best_match_num_matches = num_matches
                best_match_avg_distance = avg_distance
        
        # Print the best match information
        # Set a threshold for the number of matches or the average distance of matches
        num_matches_threshold = 10
        avg_distance_threshold = 40.0
            
        # if num_matches < num_matches_threshold or avg_distance > avg_distance_threshold:
        #     print("No match")
        # else:
        print(f"Best match: {best_match_file} with {best_match_num_matches} matches and average distance {best_match_avg_distance:.2f}")

        if best_match_file == get_ground_truth_output_name(input_image_name):
            counter += 1
            print(f'correct match {counter}')
        else:
            print('incorrect match')
            
        print(f'correct percentage{counter/200}')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        



if __name__ == '__main__':
    main()
