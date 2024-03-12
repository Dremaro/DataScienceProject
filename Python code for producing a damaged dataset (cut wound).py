import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defining a Function to apply the cut wounds algorithm on an image (then we apply this function on all images in dataset)

def apply_cut_wounds(image):
    
    # Converting the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying an intense Gaussian blurring filter with a large kernel size
    blurred_image = cv2.GaussianBlur(gray_image, (75, 75), 0)

    # Thresholding the blurred image to binarize it and creating a mask for estimating the contour
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Finding contours in the thresholded image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_copy = image.copy() # Creating a copy of the original image for applying the mask

    # Drawing contours on the copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

    # Drawing a line inside the detected contour as a simulation for the cut wound damage
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        contour_points = largest_contour.squeeze(axis=1)
        min_distance = 50  # Minimum distance between selected points
        valid_points = False
        while not valid_points:
            random_index = np.random.choice(len(contour_points), 2, replace=False)
            line_start_x, line_start_y = contour_points[random_index[0]]
            line_end_x, line_end_y = contour_points[random_index[1]]
            # Calculating the distance between the selected points
            distance = np.sqrt((line_end_x - line_start_x)**2 + (line_end_y - line_start_y)**2)
            if distance >= min_distance:
                valid_points = True
                
        # Drawing the line inside the detected contour
        cv2.line(image, (line_start_x, line_start_y), (line_end_x, line_end_y), (255, 255, 255), thickness=10)

    return image

# Now we define a main function to process all images in the dataset (according to their group) and then storing the output in a dedicated directory

def process_dataset(input_dir, output_dir):
    
    # Creating the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each image in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.tif'):  # Process only TIF images (I had to incorporate this because there was a problem otherwise)
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # loading the image
            image = cv2.imread(input_path)
            if image is not None:
                # Applying the cut wounds algorithm on the loaded image
                result_image = apply_cut_wounds(image)

                # Saving the result image
                cv2.imwrite(output_path, result_image)
            else:
                print(f"Failed to read image: {file_name}")

# Defining the input and output directories
input_directory = r'./dataset_healthy_tests/DB1_B' # To be changed in your computer
output_directory = r'./dataset_healthy_tests/Output' # To be changed in your computer

# Calling the main function to process the dataset and giving us the results
process_dataset(input_directory, output_directory)
